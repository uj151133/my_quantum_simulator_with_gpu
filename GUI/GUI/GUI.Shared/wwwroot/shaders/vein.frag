/*
Veda.js標準のフラグメントシェーダー
*/

#define RAYMARCH_SURFACE_DIST 0.0001
#define RAYMARCH_MAX_STEPS 256
#define RAYMARCH_MAX_DIST 200.0
#define iTime mod(time, 60.0)
#define TUNNEL_AMOUNT 4.0
#define ROTSPEED 0.7

precision highp float;
uniform float time;
uniform vec2 resolution;
uniform float volume;

vec3 lp1 = vec3(0.0);
float l1a = 0.0;

float c1a = 0.0;
float c2a = 0.0;
float c3a = 0.0;
float s1a = 0.0;

float mov = 0.0;
// This creates repetitions of something along the way
vec3 repeat(vec3 p, vec3 s) {
    return (fract(p / s - 0.5) - 0.5) * s;
}

vec2 repeat(vec2 p, vec2 s) {
    return (fract(p / s - 0.5) - 0.5) * s;
}

float repeat(float p, float s) {
    return (fract(p / s - 0.5) - 0.5) * s;
}

//Rotation function
mat2 rotate(float a) {
    float s = sin(a), c = cos(a);
    return mat2(c, -s, s, c);
}

vec3 tunnel(vec3 p) {
    vec3 off = vec3(0.0);
    float dd = (p.z * 0.0075);
    dd = floor(dd * 1.0) + smoothstep(0.0, 1.0, smoothstep(0.0, 1.0, fract(dd * 1.0)));
    dd *= 20.1;
    dd += iTime * 0.1;
    off.x += sin(dd) * TUNNEL_AMOUNT;
    off.y = sin(dd * 0.7) * TUNNEL_AMOUNT;
    return off;
}

vec3 navigate(vec3 p) {
    p += tunnel(p);
    p.xy *= rotate((p.z * ROTSPEED) + (iTime * ROTSPEED));
    p.z += iTime;
    p.y -= 0.3;
    return p;
}

float sdCylinder(vec2 p, float r) {
    return length(p) - r;
}

// From Inigo Quilez website, thanks!
float smin(float a, float b, float h) {
    float k = clamp((a - b) / h * 0.5 + 0.5, 0.0, 1.0);
    return mix(a, b, k) - k * (1.0 - k) * h;
}

float sdGyroid(vec3 p, float scale, float thickness, float bias, float lx, float ly) {
    vec3 p2 = p;
    p2.z = mod(p2.z, 1000.0);
    p2 *= scale;
    float ls = max(lx, ly) * 1.6;
    float gyroid  = abs(dot(sin(p2 * lx), cos(p2.zxy * ly)) - bias) / (scale * ls) - thickness;
    return gyroid;
}

float sdSphere(vec3 p, float r) {
    return length(p) - r;
}

float getDist(vec3 p) {

    p.xz *= rotate(p.z * 0.0001);

    vec3 p2 = p;
    p2.xy = repeat(p2.xy, vec2(32.0));
    p2 += tunnel(p2);
    float g1 = sdGyroid(p2, 10.83465, 0.03, 0.3, sin(iTime * 0.0045623) * 0.5 + 0.5, cos(iTime * 0.0025623) * 0.5 + 0.5);
    float g2 = sdGyroid(p2, 20.83465, 0.03, 0.3, 1.0, 1.0);
    float g3 = sdGyroid(p2, 40.83465, 0.03, 0.3, 1.0, 1.0);
    float g4 = sdGyroid(p2, 80.83465, 0.03, 0.3, 1.0, 1.0);
    float g5 = sdGyroid(p2, 160.83465, 0.03, 0.3, 1.0, 1.0);

    float c1 = sdCylinder(p2.xy, 1.0);
    c1 -= g1;
    // c1 -= g2;
    c1a += 0.02 / (0.02 + c1);
    float d = c1;

    vec3 p3 = p2;
    p3.x += sin((iTime * 1.0) + p3.z * 0.42589) * 2.0;
    p3.y += cos((iTime * 1.0) + p3.z * 0.23652) * 2.0;
    float c2 = sdCylinder(p3.xy, 0.1);
    c2 -= g1 * 0.5;
    c2a += 0.02 / (0.02 + c2);
    d = smin(d, c2, 1.0);

    vec3 p4 = p2;
    p4.x += sin(((iTime * 1.0) + p4.z * 0.42589) + 3.1415) * 2.0;
    p4.y += cos(((iTime * 1.0) + p4.z * 0.23652) + 3.1415) * 2.0;
    float c3 = sdCylinder(p4.xy, 0.1);
    c3 -= g1 * 0.5;
    c3a += 0.02 / (0.02 + c3);
    d = smin(d, c3, 1.0);

    vec3 p5 = p2;
    p5.z -= mov * 4.0;
    p5.z = repeat(p5.z, 6.0);
    float s = sdSphere(p5, 1.5);

    s += g1;
    // s += g2;
    // s += g3;
    // s += g4;
    // s += g5;

    s1a += 0.2 / (0.2 + s);
    d = smin(d, s, 1.0);

    d *= 0.5;

    return d;
}

float rayMarch(vec3 ro, vec3 rd) {
    float dO = RAYMARCH_SURFACE_DIST;

    for (int i = 0; i< RAYMARCH_MAX_STEPS; i++) {
        vec3 p = ro + rd * dO;
        float dS = getDist(p);
        dO += dS;
        if (dO > RAYMARCH_MAX_DIST || dS < RAYMARCH_SURFACE_DIST) break;
    }
    return dO;
}

vec3 getNormal(vec3 p) {
    float asd = 0.0;
    float d = getDist(p);
    vec2 e = vec2(0.01, 0.0);

    vec3 n = d - vec3(
        getDist(p - e.xyy),
        getDist(p - e.yxy),
        getDist(p - e.yyx)
    );
    return normalize(n);
}

float getLight(vec3 p, vec3 lightPos) {
    // vec3 lightPos = verc3(0.0, 0.0, 0.0);
    vec3 l = normalize(lightPos - p);
    vec3 n = getNormal(p);

    float dif = dot(n, l);

    return dif;
}

void main() {
    vec2 uv = (gl_FragCoord.xy - 0.5 * resolution.xy) / resolution.y;

    mov += iTime * 4.0;

    vec3 ro = vec3(0.0, 0.0, -10.0);
    ro.z += mov;
    ro += tunnel(ro) * 4.0;
    vec3 ta = vec3(0.0);
    ta.z += mov;
    ta += tunnel(ta) * 4.0;
    vec3 ww = normalize(ta - ro);
    vec3 uu = normalize(cross(ww, vec3(0.0, 1.0, 0.0)));
    vec3 vv = normalize(cross(uu, ww));
    vec3 rd = normalize(uv.x * uu + uv.y * vv + ww * 2.0);

    lp1 = vec3(sin(iTime * 2.9632) * 4.0, cos(iTime * 2.1452) * 3.24145, (cos(iTime * 3.1452) * 0.5 + 0.5) * 20.0);

    float d = rayMarch(ro, rd);

    vec3 p = ro + rd * d;

    float dif = getLight(lp1, p);

    vec3 col = vec3(0.0);
    col += c1a * vec3(0.2, 0.4, 1.0) * volume * 0.2;
    col += c2a * vec3(1.0, 0.15, 0.15) * volume * 0.2;
    col += c3a * vec3(1.0, 0.15, 0.15) * volume * 0.2;
    col += s1a * vec3(1.0, 0.15, 0.15) * volume * 0.2;

    // col *= smoothstep(200.0, 100.0, d);

    float ambient = volume > 0.1 ? 1.0 : smoothstep(0.1, 0.0, volume);
    col = mix(col, vec3(0.2, 0.05, 0.05) * ambient, smoothstep(75.0, 200.0, d));

    gl_FragColor = vec4(col, 1.0);
}