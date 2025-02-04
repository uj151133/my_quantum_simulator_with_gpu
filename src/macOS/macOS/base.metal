#include <metal_stdlib>
using namespace metal;

struct Complex {
    float real;
    float imag;
    
    inline Complex(float r, float i) : real(r), imag(i) {}
    
    inline Complex(float r) : real(r), imag(0.0f) {}
    
    inline Complex operator+(Complex other) const {
            return Complex(real + other.real, imag + other.imag);
    }
    
    inline Complex operator-(Complex other) const {
            return Complex(real - other.real, imag - other.imag);
    }

    inline Complex operator*(float scalar) const {
        return Complex(real * scalar, imag * scalar);
    }
    
    inline Complex operator/(float scalar) const {
        return Complex(real / scalar, imag / scalar);
    }
    
    inline Complex operator-() const {
        return Complex(-real, -imag);
    }
};

inline Complex operator*(float scalar, Complex z) {
    return Complex{scalar * z.real, scalar * z.imag};
}

struct ComplexMatrix {
    uint rows;
    uint cols;
    device Complex* data;
};

//struct ComplexMatrix2x2 {
//    Complex elements[2][2];
//};
//
//struct ComplexMatrix4x4 {
//    Complex elements[4][4];
//};



#define i Complex(0.0f, 1.0f)
#define ONE Complex(1.0f)
#define ZERO Complex(0.0f)

#define PI 3.1415926
