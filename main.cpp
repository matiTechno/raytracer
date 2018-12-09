#include <stdio.h>
#include <assert.h>
#include <vector>
#include <math.h>

#define PI 3.14159265359f

template<typename T>
struct tvec2;

template<typename T>
struct tvec3
{
    tvec3() = default;
    explicit tvec3(T v) : x(v), y(v), z(v) {}
    tvec3(T x, T y, T z) : x(x), y(y), z(z) {}
    tvec3(T x, const tvec2<T>& v);
    tvec3(const tvec2<T>& v, T z);

    template<typename U>
    explicit tvec3(tvec3<U> v) : x(v.x), y(v.y), z(v.z) {}

    tvec3& operator+=(tvec3 v) { x += v.x; y += v.y; z += v.z; return *this; }
    tvec3& operator-=(tvec3 v) { x -= v.x; y -= v.y; z -= v.z; return *this; }
    tvec3& operator*=(tvec3 v) { x *= v.x; y *= v.y; z *= v.z; return *this; }
    tvec3& operator*=(T v)     { x *= v; y *= v; z *= v; return *this; }
    tvec3& operator/=(tvec3 v) { x /= v.x; y /= v.y; z /= v.z; return *this; }
    tvec3& operator/=(T v)     { x /= v; y /= v; z /= v; return *this; }

    tvec3 operator+(tvec3 v) const { return { x + v.x, y + v.y, z + v.z }; }
    tvec3 operator-(tvec3 v) const { return { x - v.x, y - v.y, z - v.z }; }
    tvec3 operator-()        const { return { -x, -y, -z }; }
    tvec3 operator*(tvec3 v) const { return { x * v.x, y * v.y, z * v.z }; }
    tvec3 operator*(T v)     const { return { x * v, y * v, z * v }; }
    tvec3 operator/(tvec3 v) const { return { x / v.x, y / v.y, z / v.z }; }
    tvec3 operator/(T v)     const { return { x / v, y / v, z / v }; }

    const T& operator[](int idx) const { return *(&x + idx); }
    T&       operator[](int idx)       { return *(&x + idx); }

    bool operator==(tvec3 v) const { return x == v.x && y == v.y && z == v.z; }
    bool operator!=(tvec3 v) const { return !(*this == v); }

    T x;
    T y;
    T z;
};

template<typename T>
struct tvec2
{
    tvec2() = default;
    explicit tvec2(T v) : x(v), y(v) {}
    tvec2(T x, T y) : x(x), y(y) {}

    template<typename U>
    explicit tvec2(tvec2<U> v) : x(v.x), y(v.y) {}

    template<typename U>
    explicit tvec2(tvec3<U> v) : x(v.x), y(v.y) {}

    tvec2& operator+=(tvec2 v) { x += v.x; y += v.y; return *this; }
    tvec2& operator-=(tvec2 v) { x -= v.x; y -= v.y; return *this; }
    tvec2& operator*=(tvec2 v) { x *= v.x; y *= v.y; return *this; }
    tvec2& operator*=(T v)     { x *= v; y *= v; return *this; }
    tvec2& operator/=(tvec2 v) { x /= v.x; y /= v.y; return *this; }
    tvec2& operator/=(T v)     { x /= v; y /= v; return *this; }

    tvec2 operator+(tvec2 v) const { return { x + v.x, y + v.y }; }
    tvec2 operator-(tvec2 v) const { return { x - v.x, y - v.y }; }
    tvec2 operator-()        const { return { -x, -y }; }
    tvec2 operator*(tvec2 v) const { return { x * v.x, y * v.y }; }
    tvec2 operator*(T v)     const { return { x * v, y * v }; }
    tvec2 operator/(tvec2 v) const { return { x / v.x, y / v.y }; }
    tvec2 operator/(T v)     const { return { x / v, y / v }; }

    const T& operator[](int idx) const { return *(&x + idx); }
    T&       operator[](int idx)       { return *(&x + idx); }

    bool operator==(tvec2 v) const { return x == v.x && y == v.y; }
    bool operator!=(tvec2 v) const { return !(*this == v); }

    T x;
    T y;
};

template<typename T>
inline tvec3<T> operator*(T scalar, tvec3<T> v) { return v * scalar; }

template<typename T>
inline tvec2<T> operator*(T scalar, tvec2<T> v) { return v * scalar; }

template<typename T>
inline tvec3<T>::tvec3(T x, const tvec2<T>& v) : x(x), y(v.x), z(v.y) {}

template<typename T>
inline tvec3<T>::tvec3(const tvec2<T>& v, T z) : x(v.x), y(v.y), z(z) {}

using ivec3 = tvec3<int>;
using vec3 = tvec3<float>;
using ivec2 = tvec2<int>;
using vec2 = tvec2<float>;

static float dot(vec3 v1, vec3 v2) { return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z; }
static float length(vec3 v)   { return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z); }
static vec3 normalize(vec3 v) { return v * (1.f / length(v)); }
static vec3 cross(vec3 v, vec3 w) { return { v.y * w.z - v.z * w.y, v.z * w.x - v.x * w.z, v.x * w.y - v.y * w.x }; }

struct mat3
{
    vec3 i;
    vec3 j;
    vec3 k;
};

static vec3 operator*(const mat3& m, vec3 v) { return v.x * m.i + v.y * m.j + v.z * m.k; }

static float toRadians(float degrees) { return degrees / 360.f * 2.f * PI; }

static void writeToFile(const char* filename, const vec3* data, const ivec2 size)
{
    FILE* file = fopen(filename, "w");
    assert(file);
    fprintf(file, "P3 %d %d 255", size.x, size.y);
    for(int i = 0; i < size.x * size.y; ++i)
    {
        for(int x = 0; x < 3; ++x)
            fprintf(file, " %d", int(data[i][x] * 255.f + 0.5f));
    }

    fclose(file);
}

struct Ray
{
    vec3 origin;
    vec3 dir;
};

struct Sphere
{
    vec3 pos;
    float radius;
};

// dir and up must be normalized
struct Camera
{
    vec3 pos = {0.f, 0.f, 2.f};
    vec3 dir = {0.f, 0.f, -1.f};
    vec3 up = {0.f, 1.f, 0.f};
    float hfovy = 45.f; // half of field of view in y-axis angle; in degrees
};

static Ray getCameraRay(const Camera& camera, ivec2 fragPos, ivec2 imageSize)
{
    float aspectRatio = float(imageSize.x) / imageSize.y;

    vec3 offset;
    offset.x = ((fragPos.x / float(imageSize.x)) * 2.f - 1.f) * aspectRatio;
    offset.y = -1.f * ((fragPos.y / float(imageSize.y)) * 2.f - 1.f);
    offset.z = 1.f / tanf(toRadians(camera.hfovy));

    vec3 right = normalize(cross(camera.dir, camera.up));
    vec3 up = cross(right, camera.dir);

    return {camera.pos, normalize(mat3{right, up, camera.dir} * offset)};
}

static bool hit(const Sphere& sphere, const Ray& ray)
{
    vec3 sphereToRay = ray.origin - sphere.pos;
    float a = dot(ray.dir, ray.dir);
    float b = 2.f * dot(sphereToRay, ray.dir);
    float c = dot(sphereToRay, sphereToRay) - sphere.radius * sphere.radius;
    float discriminant = b * b - 4.f * a * c;
    return discriminant >= 0.f;
}

int main()
{
    std::vector<vec3> fragments;
    ivec2 imageSize = {1920, 1080};
    fragments.resize(imageSize.x * imageSize.y);

    for(vec3& v: fragments)
        v = vec3(0.f);

    Camera camera;

    // to be sure...
    camera.dir = normalize(camera.dir);
    camera.up = normalize(camera.up);

    Sphere sphere;
    sphere.pos = vec3(0.f, 0.f, -3.f);
    sphere.radius = 1.5f;

    for(int y = 0; y < imageSize.y; ++y)
    {
        for(int x = 0; x < imageSize.x; ++x)
        {
            Ray ray = getCameraRay(camera, {x, y}, imageSize);

            if(hit(sphere, ray))
                fragments[y * imageSize.x + x] = vec3(1.f, 0.f, 0.f);
        }
    }

    // tone mapping
    // ...

    // convert to sRGB
    for(vec3& v: fragments)
    {
        for(int i = 0; i < 3; ++i)
            v[i] = powf(v[i], 1.f / 2.2f);
    }

    writeToFile("render.ppm", fragments.data(), imageSize);
    return 0;
}
