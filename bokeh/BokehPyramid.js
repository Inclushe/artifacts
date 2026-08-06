/**
 * BokehPyramid.js — zero-dependency WebGL2
 * ----------------------------------------------------------------------------
 * Cinematic uniform-CoC bokeh for STATIC 2D images, at 60fps on mobile.
 *
 * The expensive scatter-as-gather disc blur is precomputed once into a pyramid
 * of blur levels (geometric radius spacing, each level stored at a resolution
 * matched to its own radius). At runtime, changing "strength" is two texture
 * reads and a mix() -- vsync-bound, not GPU-bound.
 *
 * Ported from the far-plane disc gather of CinematicDOF.fx by Frans Bouma
 * (Otis_Inf), https://github.com/FransBouma/OtisFX -- BSD 3-clause.
 * With a uniform circle of confusion the depth machinery (CoC pass, min-CoC
 * tiles, CoC gaussians, near-plane pass, combiner) collapses away and the
 * per-sample weight reduces to pure ring geometry. What survives -- and what
 * actually produces "strong" bokeh -- is the inverse-Reinhard highlight
 * expansion, the ring sampling pattern, and the busy-factor ring weighting.
 *
 * Highlights are expanded to pseudo-HDR *before* the downsample chain is built,
 * so a clipped 1-pixel specular still carries its energy into the low-res
 * levels instead of being averaged into nothing.
 *
 *   const bokeh = new BokehPyramid(canvas);
 *   bokeh.setSource(imageOrCanvas);
 *   function frame() {
 *     bokeh.update();          // builds one step per frame
 *     bokeh.setStrength(r);    // radius in full-res source pixels
 *     bokeh.render();          // to the canvas
 *   }
 *
 * Requires WebGL2 with EXT_color_buffer_float (or _half_float). Degrades to a
 * normalized 8-bit chain with a console warning otherwise.
 * ----------------------------------------------------------------------------
 */

// Cinematic Depth of Field shader, using scatter-as-gather for ReShade 3.x+
// By Frans Bouma, aka Otis / Infuse Project (Otis_Inf)
// https://fransbouma.com
//
// This shader has been released under the following license:
//
// Copyright (c) 2018-2022 Frans Bouma
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice, this
//   list of conditions and the following disclaimer.
//
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

const TAU = 6.283185307179586;
const POINTS_FIRST_RING = 7;

/* -------------------------------------------------------------------------- */
/* GL helpers                                                                  */
/* -------------------------------------------------------------------------- */

function compileShader(gl, type, src) {
	const sh = gl.createShader(type);
	gl.shaderSource(sh, src);
	gl.compileShader(sh);
	if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
		const log = gl.getShaderInfoLog(sh) || "";
		// Point at the offending line -- these shaders are generated, so a raw
		// "ERROR: 0:214" is otherwise useless.
		const line = parseInt((log.match(/\d+:(\d+)/) || [])[1], 10);
		const src_ = src.split("\n");
		const ctx = Number.isFinite(line)
			? src_
					.slice(Math.max(0, line - 4), line + 3)
					.map((l, i) => `${Math.max(1, line - 3) + i}| ${l}`)
					.join("\n")
			: src_.slice(0, 20).join("\n");
		gl.deleteShader(sh);
		throw new Error(`[BokehPyramid] shader compile failed:\n${log}\n${ctx}`);
	}
	return sh;
}

function linkProgram(gl, vsSrc, fsSrc) {
	const vs = compileShader(gl, gl.VERTEX_SHADER, vsSrc);
	const fs = compileShader(gl, gl.FRAGMENT_SHADER, fsSrc);
	const p = gl.createProgram();
	gl.attachShader(p, vs);
	gl.attachShader(p, fs);
	gl.linkProgram(p);
	gl.deleteShader(vs);
	gl.deleteShader(fs);
	if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
		const log = gl.getProgramInfoLog(p);
		gl.deleteProgram(p);
		throw new Error(`[BokehPyramid] program link failed: ${log}`);
	}
	return p;
}

/** A shader program plus a lazily-populated uniform location cache. */
class Pass {
	constructor(gl, fragSrc) {
		this.gl = gl;
		this.prog = linkProgram(gl, VERT_SRC, fragSrc);
		this._loc = new Map();
	}

	_u(name) {
		if (!this._loc.has(name)) {
			this._loc.set(name, this.gl.getUniformLocation(this.prog, name));
		}
		return this._loc.get(name);
	}

	use() {
		this.gl.useProgram(this.prog);
		return this;
	}
	f(n, v) {
		this.gl.uniform1f(this._u(n), v);
		return this;
	}
	v2(n, x, y) {
		this.gl.uniform2f(this._u(n), x, y);
		return this;
	}
	m2(n, a) {
		this.gl.uniformMatrix2fv(this._u(n), false, a);
		return this;
	}

	tex(n, texture, unit) {
		const gl = this.gl;
		gl.activeTexture(gl.TEXTURE0 + unit);
		gl.bindTexture(gl.TEXTURE_2D, texture);
		gl.uniform1i(this._u(n), unit);
		return this;
	}

	dispose() {
		this.gl.deleteProgram(this.prog);
	}
}

/** Colour texture + framebuffer. */
function makeTarget(gl, w, h, half) {
	w = Math.max(1, Math.round(w));
	h = Math.max(1, Math.round(h));

	const tex = gl.createTexture();
	gl.bindTexture(gl.TEXTURE_2D, tex);
	gl.texImage2D(
		gl.TEXTURE_2D,
		0,
		half ? gl.RGBA16F : gl.RGBA8,
		w,
		h,
		0,
		gl.RGBA,
		half ? gl.HALF_FLOAT : gl.UNSIGNED_BYTE,
		null,
	);
	gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
	gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
	// Mirrored edges: at a large uniform radius every border pixel pulls from
	// off-image. Clamp smears streaks inward, zero-fill gives a dark vignette.
	gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.MIRRORED_REPEAT);
	gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.MIRRORED_REPEAT);

	const fbo = gl.createFramebuffer();
	gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
	gl.framebufferTexture2D(
		gl.FRAMEBUFFER,
		gl.COLOR_ATTACHMENT0,
		gl.TEXTURE_2D,
		tex,
		0,
	);

	const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
	gl.bindFramebuffer(gl.FRAMEBUFFER, null);
	if (status !== gl.FRAMEBUFFER_COMPLETE) {
		throw new Error(
			`[BokehPyramid] incomplete framebuffer (0x${status.toString(16)}) at ${w}x${h}`,
		);
	}

	return {
		tex,
		fbo,
		width: w,
		height: h,
		dispose() {
			gl.deleteTexture(tex);
			gl.deleteFramebuffer(fbo);
		},
	};
}

/**
 * Draw the fullscreen triangle.
 *
 * No clear: the triangle covers every pixel of the viewport (and of the
 * scissor box when one is set), so there is nothing to clear.
 *
 * @param scissor optional [x, y, w, h] limiting which pixels are written, so a
 *   single expensive gather can be spread across several frames.
 */
function draw(gl, target, scissor) {
	gl.bindFramebuffer(gl.FRAMEBUFFER, target ? target.fbo : null);
	const w = target ? target.width : gl.drawingBufferWidth;
	const h = target ? target.height : gl.drawingBufferHeight;
	gl.viewport(0, 0, w, h);

	if (scissor) {
		gl.enable(gl.SCISSOR_TEST);
		gl.scissor(scissor[0], scissor[1], scissor[2], scissor[3]);
	}

	gl.drawArrays(gl.TRIANGLES, 0, 3);

	if (scissor) gl.disable(gl.SCISSOR_TEST);
}

/* -------------------------------------------------------------------------- */
/* shared GLSL                                                                 */
/* -------------------------------------------------------------------------- */

/** Attribute-less fullscreen triangle: 0->(-1,-1), 1->(3,-1), 2->(-1,3). */
const VERT_SRC = `#version 300 es
out vec2 vUv;
void main() {
  vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
  vUv = p;
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
`;

const FRAG_HEAD = `#version 300 es
precision highp float;
precision highp sampler2D;
in vec2 vUv;
out vec4 fragColor;
`;

/**
 * ConeOverlap is a symmetric 3x3 desaturation matrix applied before the
 * highlight expansion. Without it a pixel like (1.0, 0.2, 0.2) blows its red
 * channel to ~200x while green stays at 0.2, and every bokeh ball fringes
 * magenta. The inverse is applied on the way back out.
 */
const COLOR_GLSL = `
const float CONE_K = 0.132;

vec3 coneOverlap(vec3 c) {
  float a = 1.0 - 2.0 * CONE_K;
  float b = CONE_K;
  return vec3(dot(c, vec3(a, b, b)), dot(c, vec3(b, a, b)), dot(c, vec3(b, b, a)));
}

vec3 coneOverlapInverse(vec3 c) {
  float d = 3.0 * CONE_K - 1.0;
  float a = (CONE_K - 1.0) / d;
  float b = CONE_K / d;
  return vec3(dot(c, vec3(a, b, b)), dot(c, vec3(b, a, b)), dot(c, vec3(b, b, a)));
}

vec3 srgbToLinear(vec3 c) {
  return mix(c / 12.92, pow((c + 0.055) / 1.055, vec3(2.4)), step(vec3(0.04045), c));
}

vec3 linearToSrgb(vec3 c) {
  c = max(c, vec3(0.0));
  return mix(c * 12.92, 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055, step(vec3(0.0031308), c));
}

// Inverse Reinhard. clampMin bounds the denominator: 0.005 -> highlights reach
// ~200x. The original uses 0.001 (~1000x); the tighter bound keeps the
// accumulator far away from half-float range on mobile.
vec3 accentuateWhites(vec3 c, float boost, float gamma, float clampMin) {
  c = pow(abs(coneOverlap(c)), vec3(gamma));
  return c / max(1.001 - boost * c, clampMin);
}

vec3 correctForWhiteAccentuation(vec3 c, float boost, float gamma) {
  vec3 t = c / (1.001 + boost * c);
  return coneOverlapInverse(pow(abs(t), vec3(1.0 / gamma)));
}

float hash12(vec2 p) {
  vec3 p3 = fract(vec3(p.xyx) * 0.1031);
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.x + p3.y) * p3.z);
}
`;

const COPY_FRAG = `${FRAG_HEAD}
uniform sampler2D uSrc;
void main() { fragColor = vec4(texture(uSrc, vUv).rgb, 1.0); }
`;

const EXPAND_FRAG = `${FRAG_HEAD}
uniform sampler2D uSrc;
uniform float uBoost, uGamma, uClamp, uHdrScale, uSrgbIn;
${COLOR_GLSL}
void main() {
  vec3 c = texture(uSrc, vUv).rgb;
  c = mix(c, srgbToLinear(c), uSrgbIn);
  fragColor = vec4(accentuateWhites(c, uBoost, uGamma, uClamp) * uHdrScale, 1.0);
}
`;

// Exact 2x2 box via four bilinear-centred taps at source half-texel offsets.
const DOWNSAMPLE_FRAG = `${FRAG_HEAD}
uniform sampler2D uSrc;
uniform vec2 uTexel;
void main() {
  vec2 o = uTexel * 0.5;
  vec3 s = texture(uSrc, vUv + vec2(-o.x, -o.y)).rgb
         + texture(uSrc, vUv + vec2( o.x, -o.y)).rgb
         + texture(uSrc, vUv + vec2(-o.x,  o.y)).rgb
         + texture(uSrc, vUv + vec2( o.x,  o.y)).rgb;
  fragColor = vec4(s * 0.25, 1.0);
}
`;

const COMPOSITE_FRAG = `${FRAG_HEAD}
uniform sampler2D uA;
uniform sampler2D uB;
uniform float uT;
void main() {
  vec3 a = texture(uA, vUv).rgb;
  vec3 b = texture(uB, vUv).rgb;
  fragColor = vec4(mix(a, b, uT), 1.0);
}
`;

/* -------------------------------------------------------------------------- */
/* gather shader generation                                                    */
/* -------------------------------------------------------------------------- */

const g = (n) => Number(n).toFixed(8);

/**
 * Emits the disc gather with the ring loop unrolled on the CPU, so every loop
 * bound is a compile-time constant and there is no trigonometry in the inner
 * loop -- the sample direction is advanced by a constant 2x2 rotation, four
 * multiplies per tap. Both matter a great deal on mobile drivers.
 */
function buildGatherFrag(rings, useShape) {
	let consts = "";
	let body = "";

	for (let ring = 0; ring < rings; ring++) {
		const pts = POINTS_FIRST_RING * (ring + 1);
		const ang = TAU / pts;
		const c = Math.cos(ang);
		const s = Math.sin(ang);

		// mat2 is column-major: col0 = (cos, sin), col1 = (-sin, cos)
		consts += `const mat2 ROT${ring} = mat2(${g(c)}, ${g(s)}, ${g(-s)}, ${g(c)});\n`;

		// Matches the original: ring i samples at (i+1) * delta, and its weight is
		// mix(i / rings, 1, busy). busy -> 0 makes outer rings dominate, which is
		// what produces defined ring bokeh rather than a soft gaussian smear.
		const ringWeightBase = ring / rings;
		const ringRadiusMul = ring + 1;
		const shapeRingDist = ((ring + 1) / rings) * 0.5;

		body += `
  {
    float w = mix(${g(ringWeightBase)}, 1.0, busy);
    float rr = ${g(ringRadiusMul)} * ringRadiusDelta;
    vec2 dir = vec2(${g(c)}, ${g(s)});
    for (int p = 0; p < ${pts}; p++) {
      vec3 tap = texture(uSrc, vUv + dir * rr * uTexel).rgb;
      float tw = w;`;

		if (useShape) {
			body += `
      vec2 sd = uShapeRot * dir;
      vec4 st = texture(uShape, sd * ${g(shapeRingDist)} + 0.5);
      float sl = dot(st.rgb, vec3(0.3, 0.59, 0.11));
      tw *= step(0.01, sl);
      tap *= st.rgb * uShapeGamma;`;
		}

		body += `
      acc += vec4(tap * tw, tw);
      dir = ROT${ring} * dir;
    }
  }`;
	}

	return `${FRAG_HEAD}
uniform sampler2D uSrc;
uniform vec2  uTexel;
uniform float uRadius;
uniform float uBusy;
uniform float uBoost, uGamma, uHdrScale;
${useShape ? "uniform sampler2D uShape;\nuniform mat2 uShapeRot;\nuniform float uShapeGamma;" : ""}

${COLOR_GLSL}

${consts}

void main() {
  float busy = clamp(1.0 - uBusy, 0.0, 1.0);
  float ringRadiusDelta = uRadius / ${g(rings)};

  vec3 center = texture(uSrc, vUv).rgb;
  // highp accumulator: expanded highlights times hundreds of taps overflows
  // mediump (max 16384) and blows out to white on Mali/Adreno.
  vec4 acc = vec4(center * busy, busy);
${body}

  vec3 hdr = acc.rgb / max(acc.w, 1e-4);
  hdr /= uHdrScale;

  vec3 ldr = correctForWhiteAccentuation(hdr, uBoost, uGamma);
  ldr = linearToSrgb(ldr);
  // Dither: large smooth bokeh gradients band badly in 8-bit.
  ldr += (hash12(gl_FragCoord.xy) - 0.5) / 255.0;

  fragColor = vec4(ldr, 1.0);
}
`;
}

function tapCount(rings) {
	return (POINTS_FIRST_RING * rings * (rings + 1)) / 2;
}

function ringsForRadius(r, density, maxRings) {
	const target = Math.PI * r * r * density;
	const n = Math.ceil(
		(-1 + Math.sqrt(1 + (8 * target) / POINTS_FIRST_RING)) / 2,
	);
	return Math.max(3, Math.min(maxRings, n));
}

/* -------------------------------------------------------------------------- */
/* BokehPyramid                                                                */
/* -------------------------------------------------------------------------- */

export class BokehPyramid {
	/**
	 * @param target an HTMLCanvasElement (a WebGL2 context is created on it) or
	 *        an existing WebGL2RenderingContext.
	 */
	constructor(target, options = {}) {
		this.options = Object.assign(
			{
				/** smallest precomputed radius, in full-res pixels. Below this the
				 *  result crossfades to the sharp source, which is a dissolve rather
				 *  than a blur ramp -- so keep this low enough to be imperceptible. */
				minRadius: 3,
				/** largest precomputed radius, in full-res pixels. */
				maxRadius: 128,
				/** geometric radius spacing between levels. ~1.25-1.35 avoids the
				 *  double-ring artifact on point highlights; 2.0 shows it clearly. */
				spacing: 1.3,
				/** every level is gathered at a resolution that puts its radius near
				 *  this many pixels, so tap count stays roughly constant. */
				workingRadius: 7,
				/** minimum downsample (as a power of two) for EVERY level, including the
				 *  smallest radii. Without this the low end is gathered at full res and
				 *  those one or two levels dominate total build cost. A radius-6 blur
				 *  has nothing above the half-res Nyquist anyway. */
				minChain: 1,
				/** MAXIMUM downsample, and the reason large radii don't look mushy.
				 *  A uniformly blurred image is *nearly* band-limited, but not at bokeh
				 *  disc rims: convolving a point highlight with a hard-edged disc leaves
				 *  a hard-edged disc, and a disc kernel's transform decays only as
				 *  f^-1.5, so real high-frequency energy survives at any radius. Storing
				 *  a radius-128 level at 1/16 res therefore smears every rim over 16
				 *  full-res pixels and the whole image reads as low resolution. */
				maxChain: 2,
				/** ring cap. Large radii held at 1/4 res need far more than the 12 that
				 *  suffices when levels are allowed to shrink without limit. */
				maxRings: 24,
				/** approximate texture-sample ceiling per build step. Levels costing
				 *  more than this are rendered in horizontal strips across successive
				 *  frames, so no single frame stalls. */
				sampleBudget: 30e6,
				/** samples per working pixel inside the disc. */
				density: 1.0,
				/** cap on the base working resolution's long edge. */
				maxBaseSize: 1280,
				/** 0 = flat uniformly weighted disc (soft). 1 = outer rings dominate
				 *  (defined ring/donut bokeh). This is the main "strength" of look. */
				busyFactor: 0.75,
				/** how far the inverse tonemap reaches. The single biggest lever on
				 *  whether highlights read as bokeh balls or as gray smear. */
				highlightBoost: 0.9,
				/** artistic gamma on the expansion. 1.0 = neutral. */
				highlightGamma: 1.0,
				/** denominator floor; 0.005 ~= 200x ceiling. */
				highlightClamp: 0.005,
				/** source is sRGB-encoded (true for any normal image). */
				srgbInput: true,
				/** build steps per update() call. 1 keeps the UI at 60fps. */
				levelsPerFrame: 1,
				/** optional WebGLTexture with a custom aperture shape. */
				shapeTexture: null,
				shapeRotation: 0.0,
				shapeGamma: 1.0,
			},
			options,
		);

		if (target instanceof WebGL2RenderingContext) {
			this.gl = target;
			this.canvas = target.canvas;
		} else {
			this.canvas = target;
			this.gl = target.getContext("webgl2", {
				alpha: false,
				antialias: false,
				depth: false,
				stencil: false,
				premultipliedAlpha: false,
				preserveDrawingBuffer: false,
				powerPreference: "high-performance",
			});
			if (!this.gl) throw new Error("[BokehPyramid] WebGL2 is required.");
		}

		this.levels = [];
		this._levelTargets = [];
		this._chainTargets = [];
		this._sharpTarget = null;
		this._gatherPasses = new Map();
		this._sourceTex = null;
		this._sourceImage = null;
		this._buildIndex = -1;
		this._buildStrip = 0;
		this._stepsDone = 0;
		this._stepsTotal = 1;
		this._radius = this.options.maxRadius;
		this._lost = false;

		if (this.canvas && this.canvas.addEventListener) {
			this.canvas.addEventListener(
				"webglcontextlost",
				(this._onLost = (e) => {
					e.preventDefault();
					this._lost = true;
				}),
			);
			this.canvas.addEventListener(
				"webglcontextrestored",
				(this._onRestored = () => {
					this._lost = false;
					this._initGL();
					if (this._sourceImage) this.setSource(this._sourceImage);
				}),
			);
		}

		this._initGL();
	}

	/* ---------------------------------------------------------------------- */

	_initGL() {
		const gl = this.gl;

		// RGBA16F as a colour attachment needs one of these.
		this.halfFloatOK = !!(
			gl.getExtension("EXT_color_buffer_float") ||
			gl.getExtension("EXT_color_buffer_half_float")
		);
		if (!this.halfFloatOK) {
			console.warn(
				"[BokehPyramid] No float render targets available. Falling " +
					"back to a normalized 8-bit HDR chain -- highlights will band.",
			);
		}
		this._hdrScale = this.halfFloatOK ? 1.0 : this.options.highlightClamp;

		// A VAO must be bound to draw, even with no attributes.
		if (this._vao) gl.deleteVertexArray(this._vao);
		this._vao = gl.createVertexArray();
		gl.bindVertexArray(this._vao);

		gl.disable(gl.DEPTH_TEST);
		gl.disable(gl.BLEND);
		gl.disable(gl.CULL_FACE);
		gl.disable(gl.SCISSOR_TEST);

		for (const p of this._gatherPasses ? this._gatherPasses.values() : [])
			p.dispose();
		this._gatherPasses = new Map();

		this._copyPass = new Pass(gl, COPY_FRAG);
		this._expandPass = new Pass(gl, EXPAND_FRAG);
		this._downPass = new Pass(gl, DOWNSAMPLE_FRAG);
		this._compositePass = new Pass(gl, COMPOSITE_FRAG);
	}

	_gatherPass(rings) {
		const useShape = !!this.options.shapeTexture;
		const key = `${rings}|${useShape ? 1 : 0}`;
		let pass = this._gatherPasses.get(key);
		if (!pass) {
			pass = new Pass(this.gl, buildGatherFrag(rings, useShape));
			this._gatherPasses.set(key, pass);
		}
		return pass;
	}

	/* ---------------------------------------------------------------------- */

	/** Plan the levels: radius, chain level to gather from, and ring count. */
	_planLevels() {
		const o = this.options;
		const radii = [];
		for (let r = o.minRadius; r < o.maxRadius; r *= o.spacing) radii.push(r);
		radii.push(o.maxRadius);

		const hardMaxC = Math.max(
			0,
			Math.floor(Math.log2(Math.min(this.baseWidth, this.baseHeight) / 8)),
		);
		const ceilC = Math.min(hardMaxC, o.maxChain);
		const floorC = Math.min(o.minChain, ceilC);

		this.levels = radii.map((radius) => {
			const c = Math.min(
				ceilC,
				Math.max(floorC, Math.round(Math.log2(radius / o.workingRadius))),
			);
			const scale = 1 << c;
			const w = Math.max(1, Math.round(this.baseWidth / scale));
			const h = Math.max(1, Math.round(this.baseHeight / scale));
			const working = radius / scale;
			const rings = ringsForRadius(working, o.density, o.maxRings);
			const taps = tapCount(rings);
			const samples = taps * w * h;
			const strips = Math.max(1, Math.ceil(samples / o.sampleBudget));
			return {
				radius,
				chain: c,
				width: w,
				height: h,
				working,
				rings,
				taps,
				samples,
				strips,
			};
		});

		this.chainDepth = this.levels.reduce((m, l) => Math.max(m, l.chain), 0);
		this._stepsTotal = 1 + this.levels.reduce((s, l) => s + l.strips, 0);
	}

	/**
	 * Point the pyramid at an image and schedule a rebuild. Accepts anything
	 * texImage2D takes: HTMLImageElement, HTMLCanvasElement, ImageBitmap, ...
	 * Call update() each frame afterwards until ready === true.
	 */
	setSource(image) {
		const gl = this.gl;
		this._disposeTargets();
		this._sourceImage = image;

		if (this._sourceTex) gl.deleteTexture(this._sourceTex);
		this._sourceTex = gl.createTexture();
		gl.bindTexture(gl.TEXTURE_2D, this._sourceTex);
		// GL's texel row 0 is the bottom of the framebuffer, so upload flipped to
		// keep the image upright all the way through the chain to the canvas.
		gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
		gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false);
		gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
		gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, image);
		gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

		const sw = image.naturalWidth || image.videoWidth || image.width || 1024;
		const sh = image.naturalHeight || image.videoHeight || image.height || 1024;
		this.sourceWidth = sw;
		this.sourceHeight = sh;
		this.aspect = sw / sh;

		const k = Math.min(1, this.options.maxBaseSize / Math.max(sw, sh));
		this.baseWidth = Math.max(8, Math.round(sw * k));
		this.baseHeight = Math.max(8, Math.round(sh * k));

		this._planLevels();

		this._sharpTarget = makeTarget(gl, this.baseWidth, this.baseHeight, false);
		this._levelTargets = this.levels.map((l) =>
			makeTarget(gl, l.width, l.height, false),
		);
		// The HDR chain is allocated lazily in _buildChain and freed again as soon
		// as the last level is gathered -- it is build-time scaffolding only, and
		// at half-float it costs more than every level combined.

		this._buildIndex = -1;
		this._buildStrip = 0;
		this._stepsDone = 0;
		return this;
	}

	/** Resize the canvas backing store. Independent of the pyramid. */
	setSize(
		cssWidth,
		cssHeight,
		dpr = Math.min(window.devicePixelRatio || 1, 2),
	) {
		if (!this.canvas) return this;
		this.canvas.width = Math.max(1, Math.round(cssWidth * dpr));
		this.canvas.height = Math.max(1, Math.round(cssHeight * dpr));
		this.canvas.style.width = `${cssWidth}px`;
		this.canvas.style.height = `${cssHeight}px`;
		return this;
	}

	/** Rebuild in place, e.g. after changing busyFactor or highlightBoost. */
	rebuild() {
		this._buildIndex = -1;
		this._buildStrip = 0;
		this._stepsDone = 0;
		return this;
	}

	get ready() {
		return this._sourceTex !== null && this._buildIndex >= this.levels.length;
	}

	get progress() {
		if (!this._sourceTex) return 0;
		if (this.ready) return 1;
		return Math.max(0, Math.min(1, this._stepsDone / this._stepsTotal));
	}

	/** Build at most levelsPerFrame steps. Call once per frame. */
	update() {
		if (!this._sourceTex || this._lost || this.ready) return false;
		for (let i = 0; i < this.options.levelsPerFrame && !this.ready; i++) {
			this._buildStep();
		}
		return !this.ready;
	}

	/** Build everything now. Blocks; only use for offline/desktop rendering. */
	buildAll() {
		while (this.update()) {
			/* keep going */
		}
		return this;
	}

	_buildStep() {
		const gl = this.gl;
		gl.bindVertexArray(this._vao);

		if (this._buildIndex < 0) {
			this._buildChain();
			this._buildIndex = 0;
			this._buildStrip = 0;
			this._stepsDone = 1;
			return;
		}

		const lvl = this.levels[this._buildIndex];
		this._gatherLevel(this._buildIndex, this._buildStrip);
		this._stepsDone++;
		this._buildStrip++;
		if (this._buildStrip >= lvl.strips) {
			this._buildIndex++;
			this._buildStrip = 0;
			if (this._buildIndex >= this.levels.length) this._freeChain();
		}
	}

	_buildChain() {
		const gl = this.gl;
		const o = this.options;

		if (this._chainTargets.length === 0) {
			for (let c = 0; c <= this.chainDepth; c++) {
				const s = 1 << c;
				this._chainTargets.push(
					makeTarget(
						gl,
						this.baseWidth / s,
						this.baseHeight / s,
						this.halfFloatOK,
					),
				);
			}
		}

		this._copyPass.use().tex("uSrc", this._sourceTex, 0);
		draw(gl, this._sharpTarget);

		this._expandPass
			.use()
			.tex("uSrc", this._sourceTex, 0)
			.f("uBoost", o.highlightBoost)
			.f("uGamma", o.highlightGamma)
			.f("uClamp", o.highlightClamp)
			.f("uHdrScale", this._hdrScale)
			.f("uSrgbIn", o.srgbInput ? 1 : 0);
		draw(gl, this._chainTargets[0]);

		for (let c = 1; c < this._chainTargets.length; c++) {
			const src = this._chainTargets[c - 1];
			this._downPass
				.use()
				.tex("uSrc", src.tex, 0)
				.v2("uTexel", 1 / src.width, 1 / src.height);
			draw(gl, this._chainTargets[c]);
		}
	}

	_gatherLevel(i, strip) {
		const gl = this.gl;
		const o = this.options;
		const lvl = this.levels[i];
		const target = this._levelTargets[i];
		const pass = this._gatherPass(lvl.rings);

		pass
			.use()
			.tex("uSrc", this._chainTargets[lvl.chain].tex, 0)
			.v2("uTexel", 1 / lvl.width, 1 / lvl.height)
			.f("uRadius", lvl.working)
			.f("uBusy", o.busyFactor)
			.f("uBoost", o.highlightBoost)
			.f("uGamma", o.highlightGamma)
			.f("uHdrScale", this._hdrScale);

		if (o.shapeTexture) {
			// The original composes R(-90) * R(rot + 270) * dir, which reduces to
			// R(rot + 180) * dir.
			const a = TAU * o.shapeRotation + Math.PI;
			const c = Math.cos(a),
				s = Math.sin(a);
			pass
				.tex("uShape", o.shapeTexture, 1)
				.m2("uShapeRot", [c, s, -s, c])
				.f("uShapeGamma", o.shapeGamma);
		}

		if (lvl.strips <= 1) {
			draw(gl, target);
			return;
		}

		const rows = Math.ceil(target.height / lvl.strips);
		const y0 = strip * rows;
		const h = Math.min(rows, target.height - y0);
		if (h <= 0) return;
		draw(gl, target, [0, y0, target.width, h]);
	}

	_freeChain() {
		for (const t of this._chainTargets) t.dispose();
		this._chainTargets = [];
	}

	/* ---------------------------------------------------------------------- */

	/** Set blur strength as a radius in full-resolution source pixels. */
	setStrength(radiusInPixels) {
		this._radius = Math.max(0, radiusInPixels);
		return this;
	}

	get strength() {
		return this._radius;
	}

	/** Which two textures bracket the current radius, and the blend factor. */
	_resolve() {
		const R = this._radius;
		const built = Math.min(this._buildIndex, this.levels.length) - 1;
		if (built < 0) {
			return { a: this._sharpTarget.tex, b: this._sharpTarget.tex, t: 0 };
		}

		if (R <= this.levels[0].radius) {
			const t = R / this.levels[0].radius;
			return { a: this._sharpTarget.tex, b: this._levelTargets[0].tex, t };
		}

		let hi = 1;
		while (hi < this.levels.length && this.levels[hi].radius < R) hi++;
		hi = Math.min(hi, built);
		const lo = Math.max(0, hi - 1);
		if (hi <= lo) {
			return {
				a: this._levelTargets[lo].tex,
				b: this._levelTargets[lo].tex,
				t: 0,
			};
		}

		// Log interpolation matches the geometric level spacing.
		const rl = Math.log(this.levels[lo].radius);
		const rh = Math.log(this.levels[hi].radius);
		const t = Math.max(
			0,
			Math.min(1, (Math.log(Math.max(R, 1e-3)) - rl) / (rh - rl)),
		);
		return { a: this._levelTargets[lo].tex, b: this._levelTargets[hi].tex, t };
	}

	/**
	 * Draw the current strength. Writes display-ready sRGB.
	 * @param target optional target from makeTarget(); defaults to the canvas.
	 */
	render(target = null) {
		if (!this._sourceTex || this._lost) return;
		const gl = this.gl;
		gl.bindVertexArray(this._vao);
		const { a, b, t } = this._resolve();
		this._compositePass.use().tex("uA", a, 0).tex("uB", b, 1).f("uT", t);
		draw(gl, target);
	}

	/* ---------------------------------------------------------------------- */

	stats() {
		let bytes = this.baseWidth * this.baseHeight * 4;
		for (const l of this.levels) bytes += l.width * l.height * 4;
		let chainBytes = 0;
		for (let c = 0; c <= this.chainDepth; c++) {
			const s = 1 << c;
			chainBytes +=
				(this.baseWidth / s) *
				(this.baseHeight / s) *
				(this.halfFloatOK ? 8 : 4);
		}
		const totalSamples = this.levels.reduce((s, l) => s + l.samples, 0);
		return {
			base: [this.baseWidth, this.baseHeight],
			levelCount: this.levels.length,
			levelMB: +(bytes / 1048576).toFixed(2),
			chainMB: +(chainBytes / 1048576).toFixed(2),
			buildSamplesM: +(totalSamples / 1e6).toFixed(1),
			buildSteps: this._stepsTotal,
			levels: this.levels.map((l) => ({
				r: +l.radius.toFixed(1),
				res: `${l.width}x${l.height}`,
				rings: l.rings,
				taps: l.taps,
				strips: l.strips,
			})),
		};
	}

	_disposeTargets() {
		for (const t of this._levelTargets) t.dispose();
		this._freeChain();
		if (this._sharpTarget) this._sharpTarget.dispose();
		this._levelTargets = [];
		this._sharpTarget = null;
	}

	dispose() {
		const gl = this.gl;
		this._disposeTargets();
		if (this._sourceTex) gl.deleteTexture(this._sourceTex);
		this._sourceTex = null;
		for (const p of this._gatherPasses.values()) p.dispose();
		this._gatherPasses.clear();
		this._copyPass.dispose();
		this._expandPass.dispose();
		this._downPass.dispose();
		this._compositePass.dispose();
		if (this._vao) gl.deleteVertexArray(this._vao);
		if (this.canvas && this.canvas.removeEventListener) {
			this.canvas.removeEventListener("webglcontextlost", this._onLost);
			this.canvas.removeEventListener("webglcontextrestored", this._onRestored);
		}
	}
}

export default BokehPyramid;
