#ifndef GATE_METAL
#define GATE_METAL

#include <metal_stdlib>
using namespace metal;

// アダマールゲートのみをサポート
enum class GateType : uint32_t {
    H = 1
    V = 2
    RX = 3
    RY = 4
};

// double-double表現（高精度浮動小数点）
struct float64_t {
    float hi;
    float lo;
    
    float64_t() : hi(0.0f), lo(0.0f) {}
    float64_t(float h) : hi(h), lo(0.0f) {}
    float64_t(float h, float l) : hi(h), lo(l) {}
    
    float64_t operator+(const float64_t& other) const {
        float s1 = hi + other.hi;
        float s2 = lo + other.lo;
        float v = s1 - hi;
        float u = s1 - v;
        float w = hi - u;
        float x = other.hi - v;
        float y = w + x;
        float z = y + s2;
        return float64_t(s1 + z, z - (s1 + z - s1));
    }
    
    // double-doubleの減算
    float64_t operator-(const float64_t& other) const {
        return *this + float64_t(-other.hi, -other.lo);
    }
    
    // double-doubleの乗算（簡易版）
    float64_t operator*(const float64_t& other) const {
        float p1 = hi * other.hi;
        float p2 = hi * other.lo + lo * other.hi;
        return float64_t(p1, p2 + lo * other.lo);
    }
    
    // 単精度との乗算
    float64_t operator*(float scalar) const {
        return float64_t(hi * scalar, lo * scalar);
    }
    
    // 符号反転
    float64_t operator-() const {
        return float64_t(-hi, -lo);
    }
    
    // double値への変換（近似）
    double to_double() const {
        return double(hi) + double(lo);
    }
    
    // float値への変換（精度低下）
    float to_float() const {
        return hi + lo;
    }
};

namespace double_constants {
    constant float64_t INV_SQRT2 = float64_t(0.7071067811865475f, 2.4400844362104849e-8f);
    
    constant float64_t PI_HALF = float64_t(1.5707963267948966f, 6.123233995736766e-17f);
    
    constant float64_t PI = float64_t(3.1415926535897932f, 3.8461045789309494e-17f);
};

struct Matrix2x2Split {
    float64_t real[2][2];
    float64_t imag[2][2];
    
    Matrix2x2Split() {
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                real[i][j] = float64_t();
                imag[i][j] = float64_t();
            }
        }
    }
};

// ハッシュテーブルのエントリ（アダマールゲート専用）
struct HadamardHashEntry {
    uint64_t hash;
    Matrix2x2Split matrix;
    bool is_valid;
    
    HadamardHashEntry() : hash(0), is_valid(false) {}
};

constant uint32_t HASH_TABLE_SIZE = 256;

inline uint64_t hash(uint32_t gate_type) {
    // アダマールゲート専用の固定ハッシュ値
    return 0x1234567890ABCDEFULL + gate_type;
}

inline Matrix2x2Split create_hadamard_matrix() {
    float64_t coe_real = double_constants::INV_SQRT2;
    float64_t coe_imag = float64_t();

    Matrix2x2Split matrix;

    matrix.real[0][0] = float64_t(1.0);
    matrix.real[0][1] = float64_t(1.0);
    matrix.real[1][0] = float64_t(1.0);
    matrix.real[1][1] = float64_t(-1.0);
    
    matrix.imag[0][0] = float64_t();
    matrix.imag[0][1] = float64_t();
    matrix.imag[1][0] = float64_t();
    matrix.imag[1][1] = float64_t();
    
    return matrix;
}

inline Matrix2x2Split create_v_matrix() {
    float64_t coe_real = float64_t(0.5);
    float64_t coe_imag = float64_t(0.5);

    Matrix2x2Split matrix;

    matrix.real[0][0] = float64_t(1.0);
    matrix.real[0][1] = float64_t();
    matrix.real[1][0] = float64_t();
    matrix.real[1][1] = float64_t(1.0);

    matrix.imag[0][0] = float64_t();
    matrix.imag[0][1] = float64_t(1.0);
    matrix.imag[1][0] = float64_t(1.0);
    matrix.imag[1][1] = float64_t();

    return matrix;
}

inline Matrix2x2Split create_rx_matrix(float64_t theta) {
    float64_t theta_half = theta / 2.0;

    float64_t coe_imag = float64_t();

    Matrix2x2Split matrix;

    matrix.real[0][0] = float64_t(1.0);
    matrix.imag[0][1] = float64_t();
    matrix.imag[1][0] = float64_t();
    matrix.real[1][1] = float64_t(1.0);

    matrix.imag[0][0] = float64_t();
    matrix.imag[1][1] = float64_t();

    return matrix;
}

inline Matrix2x2Split create_ry_matrix(float64_t theta) {
    float64_t theta_half = theta / 2.0;

    float64_t coe_imag = float64_t();

    Matrix2x2Split matrix;

    matrix.real[0][0] = float64_t(1.0);
    matrix.real[1][1] = float64_t(1.0);

    matrix.imag[0][0] = float64_t();
    matrix.imag[0][1] = float64_t();
    matrix.imag[1][0] = float64_t();
    matrix.imag[1][1] = float64_t();

    return matrix;
}

inline HadamardHashEntry create_hadamard_entry() {
    HadamardHashEntry entry;
    entry.hash = hash(static_cast<uint32_t>(GateType::H));
    entry.matrix = create_h_matrix();
    entry.is_valid = true;
    return entry;
}

#endif