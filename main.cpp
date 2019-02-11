#include <stdio.h>
#include <assert.h>
#include <vector>
#include <math.h>
#include <float.h>
#include <random>
#include <memory>
#include <pthread.h>
#include <string.h>
#include <sys/sysinfo.h>

#define PI 3.14159265359f

#define FORCE_INLINE __attribute__((always_inline)) inline

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

template<typename T>
T max(T a, T b) { return a > b ? a : b; }

template<typename T>
T min(T a, T b) { return a < b ? a : b; }

FORCE_INLINE static float dot(vec3 v1, vec3 v2) { return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z; }
FORCE_INLINE static float length(vec3 v)   { return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z); }
FORCE_INLINE static vec3 normalize(vec3 v) { return v * (1.f / length(v)); }
FORCE_INLINE static vec3 cross(vec3 v, vec3 w) { return { v.y * w.z - v.z * w.y, v.z * w.x - v.x * w.z, v.x * w.y - v.y * w.x }; }

// n must be normalized
FORCE_INLINE static vec3 reflect(vec3 toReflect, vec3 n) { return dot(toReflect, n) * n * -2.f + toReflect; }

/*
// ior - index of refraction
static vec3 refract(vec3 toRefract, vec3 n, float ior)
{

}
*/

struct mat3
{
    vec3 i;
    vec3 j;
    vec3 k;
};

FORCE_INLINE static vec3 operator*(const mat3& m, vec3 v) { return v.x * m.i + v.y * m.j + v.z * m.k; }

FORCE_INLINE static float toRadians(float degrees) { return degrees / 360.f * 2.f * PI; }

static __thread std::mt19937* _rng = nullptr;

FORCE_INLINE static float random01()
{
    assert(_rng);
    static __thread std::uniform_real_distribution<float> d(0.f, 1.f);
    return d(*_rng);
}

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

FORCE_INLINE static vec3 unitSphereSample()
{
    vec3 p(1.f);

    while(length(p) > 1.f)
    {
        p = 2.f * vec3(random01(), random01(), random01()) - vec3(1.f);
    }

    return p;
}

// dir must be normalized
struct Ray
{
    vec3 origin;
    vec3 dir;
};

// dir and up must be normalized
struct Camera
{
    vec3 pos = {0.f, 0.f, 2.f};
    vec3 dir = {0.f, 0.f, -1.f};
    vec3 up = {0.f, 1.f, 0.f};
    float hfovy = 45.f; // half of field of view in y-axis angle; in degrees
};

FORCE_INLINE static Ray getCameraRay(const Camera& camera, vec2 fragPos, ivec2 imageSize)
{
    float aspectRatio = float(imageSize.x) / imageSize.y;

    // in camera space
    vec3 sensorPos;
    sensorPos.x = ((fragPos.x / imageSize.x) * 2.f - 1.f) * aspectRatio;
    sensorPos.y = -1.f * ((fragPos.y / imageSize.y) * 2.f - 1.f);
    sensorPos.z = 1.f / tanf(toRadians(camera.hfovy));

    vec3 right = normalize(cross(camera.dir, camera.up));
    vec3 up = cross(right, camera.dir);

    //                  convert to world coordinates
    return {camera.pos, normalize(mat3{right, up, camera.dir} * sensorPos)};
}

struct ScatterData
{
    // input
    Ray inputRay;
    vec3 point;
    vec3 normal;
    // output
    Ray outputRay;
    vec3 attenuation;
};

struct Material
{
    // returns false if ray is absorbed
    virtual bool scatter(ScatterData& sdata) = 0;
};

struct Collision
{
    float distance;
    vec3 point;
    vec3 normal;
    Material* material;
};

struct Sphere
{
    vec3 pos;
    float radius;
    Material* material;
};

FORCE_INLINE static bool collides(const Ray& ray, const Sphere& sphere, Collision& collision, float minDistance, float maxDistance)
{
    vec3 sphereToRay = ray.origin - sphere.pos;
    float a = dot(ray.dir, ray.dir);
    float b = 2.f * dot(sphereToRay, ray.dir);
    float c = dot(sphereToRay, sphereToRay) - sphere.radius * sphere.radius;
    float discriminant = b * b - 4.f * a * c;

    if(discriminant > 0.f)
    {
        for(int i = 0; i < 2; ++i)
        {
            int sign = i * 2 - 1;
            float root = (-b + sign * sqrtf(discriminant)) / 2.f;

            if(root >= minDistance && root <= maxDistance)
            {
                collision.distance = root;
                collision.point = ray.origin + ray.dir * root;
                collision.normal = normalize(collision.point - sphere.pos);
                collision.material = &*sphere.material;
                return true;
            }
        }
    }
    return false;
}

static vec3 getRayColor(const Ray& ray, int depth, const std::vector<Sphere>& spheres)
{
    float maxDistance = FLT_MAX;
    Collision collision;

    for(const Sphere& sphere: spheres)
    {
        if(collides(ray, sphere, collision, 0.001f, maxDistance))
            maxDistance = collision.distance;
    }

    if(maxDistance != FLT_MAX)
    {
        ScatterData sdata;
        sdata.inputRay = ray;
        sdata.point = collision.point;
        sdata.normal = collision.normal;

        if(depth < 9 && collision.material->scatter(sdata))
            return sdata.attenuation * getRayColor(sdata.outputRay, depth + 1, spheres);

        return vec3(0.f);
    }

    // background
    float t = (ray.dir.y + 1.f) / 2.f;
    vec3 colorBot(1.f);
    vec3 colorTop(0.5f, 0.7f, 1.f);
    return (1.f - t) * colorBot + t * colorTop;
}

struct Lambertian: public Material
{
    bool scatter(ScatterData& sdata) override
    {
        vec3 target = sdata.point + sdata.normal + unitSphereSample();
        vec3 dir = normalize(target - sdata.point);
        sdata.outputRay = {sdata.point, dir};
        sdata.attenuation = albedo;
        return true;
    }

    vec3 albedo;
};

struct Metal: public Material
{
    bool scatter(ScatterData& sdata) override
    {
        sdata.outputRay = {sdata.point,
                           normalize( reflect(sdata.inputRay.dir, sdata.normal) + fuzz * unitSphereSample() )};

        sdata.attenuation = albedo;
        return dot(sdata.normal, sdata.outputRay.dir) > 0.f;
    }

    vec3 albedo;
    float fuzz = 0.f;
};

/*
struct Dielectric: public Material
{
    bool scatter(ScatterData& sdata) override
    {
    }
};
*/

static void initScene(Camera& camera, std::vector<Sphere>& spheres, std::vector<std::unique_ptr<Material>>& materials)
{
    camera.pos = vec3(0.f, 2.f, 3.f);
    camera.dir = normalize(vec3(0.f) - camera.pos);
    // to be sure...
    camera.up = normalize(camera.up);

    // materials
    {
        Lambertian mat;
        mat.albedo = vec3(0.8f, 0.f, 0.f);
        materials.push_back(std::make_unique<Lambertian>(mat));
    }
    {
        Metal mat;
        mat.albedo = vec3(0.6f);
        materials.push_back(std::make_unique<Metal>(mat));
    }
    {
        Lambertian mat;
        mat.albedo = vec3(0.8f, 0.3f, 0.3f);
        materials.push_back(std::make_unique<Lambertian>(mat));
    }
    {
        Metal mat;
        mat.albedo = vec3(0.8f, 0.6f, 0.2f);
        mat.fuzz = 0.3f;
        materials.push_back(std::make_unique<Metal>(mat));
    }

    // spheres
    {
        Sphere sphere;
        sphere.radius = 100.f;
        sphere.pos = vec3(0.f, -100.f, 0.f);
        sphere.material = &*materials[0];
        spheres.push_back(sphere);
    }
    {
        Sphere sphere;
        sphere.radius = 1.5f;
        sphere.pos = vec3(0.f, 1.f, -3.f);
        sphere.material = &*materials[1];
        spheres.push_back(sphere);
    }
    {
        Sphere sphere;
        sphere.radius = 1.f;
        sphere.pos = vec3(-4.f, 1.1f, -2.f);
        sphere.material = &*materials[2];
        spheres.push_back(sphere);
    }
    {
        Sphere sphere;
        sphere.radius = 1.f;
        sphere.pos = vec3(3.5f, 1.1f, -2.f);
        sphere.material = &*materials[3];
        spheres.push_back(sphere);
    }

    // x - start, y - size (this is not intuitive)
    vec2 xrange(-3.f, 6.f);
    vec2 yrange(0.5f, 0.5f);
    vec2 zrange(-3.f, 5.f);
    vec2 radiusRange(0.05f, 0.1f);

    for(int i = 0; i < 100; ++i)
    {
        Sphere sphere;
        sphere.radius = radiusRange.x + radiusRange.y * random01();
        sphere.pos = vec3(xrange.x, yrange.x, zrange.x) + vec3(random01() * xrange.y, random01() * yrange.y, random01() * zrange.y);
        sphere.material = random01() > 0.5f ? &*materials[1] : &*materials[3];
        spheres.push_back(sphere);
    }
}

struct JobData
{
    ivec2 renderSize;
    vec3* fragments;
    int samplesPerFrag;
    std::vector<Sphere>* spheres;
    Camera* camera;
};

// it seems that this is not affecting anything... I don't know why
#define PX_CHUNK_SIZE 16

static int _progress = 0;

static void* job(void* data)
{
    std::mt19937 rng;
    _rng = &rng;
    {
        std::random_device rd;
        rng.seed(rd());
    }

    vec3 fragmentsLocal[PX_CHUNK_SIZE];
    JobData* jobData = (JobData*)data;
    const int pixelCount = jobData->renderSize.x * jobData->renderSize.y;

    while(true)
    {
        int start = __sync_fetch_and_add(&_progress, PX_CHUNK_SIZE);

        if(start >= pixelCount)
            break;

        const int pixelsToRender = min(PX_CHUNK_SIZE, pixelCount - start);

        for(int i = 0; i < pixelsToRender; ++i)
        {
            ivec2 pixelPos = {(start + i) % jobData->renderSize.x, (start + i) / jobData->renderSize.x};
            vec3 fragColor(0.f);

            for(int s = 0; s < jobData->samplesPerFrag; ++s)
            {
                const Ray ray = getCameraRay(*jobData->camera, vec2(pixelPos) + vec2(random01(), random01()), jobData->renderSize);
                fragColor += getRayColor(ray, 0, *jobData->spheres);
            }

            fragmentsLocal[i] = fragColor / jobData->samplesPerFrag;
        }

        for(int it = 0; it < pixelsToRender; ++it)
        {
            vec3& v = fragmentsLocal[it];

            // temporary, clamp color to 1.f (replace with tone mapping)
            for(int i = 0; i < 3; ++i)
                v[i] = min(v[i], 1.f);

            // convert to sRGB
            for(int i = 0; i < 3; ++i)
                v[i] = powf(v[i], 1.f / 2.2f);
        }

        memcpy(jobData->fragments + start, fragmentsLocal, pixelsToRender * sizeof(vec3));
    }

    return nullptr;
}

int main(int argc, const char**)
{
    std::mt19937 rng;
    _rng = &rng;
    {
        std::random_device rd;
        rng.seed(rd());
    }

    std::vector<vec3> fragments;
    ivec2 imageSize = {1920, 1080};
    int perFragSamples = argc > 1 ? 100 : 10;
    fragments.resize(imageSize.x * imageSize.y);

    for(vec3& v: fragments)
        v = vec3(0.f);

    Camera camera;
    std::vector<Sphere> spheres;
    std::vector<std::unique_ptr<Material>> materials;

    initScene(camera, spheres, materials);

    JobData jobData;
    jobData.renderSize = imageSize;
    jobData.fragments = fragments.data();
    jobData.samplesPerFrag = perFragSamples;
    jobData.spheres = &spheres;
    jobData.camera = &camera;

    pthread_t threads[get_nprocs()];

    for(pthread_t& thread: threads)
    {
        int r = pthread_create(&thread, nullptr, job, &jobData);
        assert(!r);
        (void)r;
    }

    for(pthread_t& thread: threads)
    {
        int r = pthread_join(thread, nullptr);
        assert(!r);
        (void)r;
    }

    writeToFile("render.ppm", fragments.data(), imageSize);
    return 0;
}
