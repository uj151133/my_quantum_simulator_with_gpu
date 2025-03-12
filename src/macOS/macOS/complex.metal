#ifndef COMPLEX
#define COMPLEX


#include <metal_stdlib>
#include <metal_math>
using namespace metal;

struct Complex {
    simd_float2 value;
//    simd_double2 value;
    
    inline Complex(float r, float i = 0.0f) : value(r, i) {}
    
    inline Complex operator+(Complex other) const {
        return Complex(value.x + other.value.x, value.y + other.value.y);
    }
    
    inline Complex operator+(float scalar) const {
            return Complex(value.x + scalar, value.y);
    }
    
    inline Complex operator-(Complex other) const {
            return Complex(value.x - other.value.x, value.y - other.value.y);
    }

    inline Complex operator*(float scalar) const {
        return Complex(value.x * scalar, value.y * scalar);
    }
    
    inline Complex operator*(Complex other) const {
        return Complex(value.x * other.value.x - value.y * other.value.y, value.x * other.value.y + value.y * other.value.x);
    }
    
    inline Complex operator/(float scalar) const {
        return Complex(value.x / scalar, value.y / scalar);
    }
    
    inline Complex operator-() const {
        return Complex(-value.x, -value.y);
    }
};

inline Complex operator*(float scalar, Complex z) {
    return Complex{scalar * z.value.x, scalar * z.value.y};
}

inline Complex operator+(float scalar, Complex z) {
    return Complex{scalar + z.value.x, z.value.y};
}

inline Complex operator-(float scalar, Complex z) {
    return Complex{scalar - z.value.x, -z.value.y};
}

struct ComplexMatrix {
    uint rows;
    uint cols;
    device Complex* data;
};

#define i Complex(0.0f, 1.0f)
#define ONE Complex(1.0f)
#define ZERO Complex(0.0f)
#define PI 3.1415926535
#endif

