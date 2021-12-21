# include<stdio.h>
#ifdef _WIN32  // 如果在 Windows 上
#define DLLEXPORT __declspec(dllexport)
#else          // 否则在 Unix 类系统上
#define DLLEXPORT __attribute__((visibility("default")))
#endif

DLLEXPORT void print_hello() {
    printf("Hello World!\n");
}
