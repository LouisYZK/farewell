# include<stdio.h>
#ifdef _WIN32  // 如果在 Windows 上
#define DLLEXPORT __declspec(dllexport)
#else          // 否则在 Unix 类系统上
#define DLLEXPORT __attribute__((visibility("default")))
#endif

extern "C" DLLEXPORT void print_hello() {
    printf("Hello World!\n");
}

extern "C" DLLEXPORT int double_int(int x) {
    return x * 2;
}

extern "C" DLLEXPORT int double_float(float x) {
    return x * 2.f;
}

extern "C" DLLEXPORT void print_str(const char* s) {
    printf(" Str is %s\n", s);
}

extern "C" DLLEXPORT void test_numpy(float* arr, size_t size) {
    for ( size_t i = 0; i < size; ++i ) {
        printf ("%ld: %f\n", i, arr[i]);
    }
}
