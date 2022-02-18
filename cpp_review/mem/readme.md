# 访存优化与CPU缓存机制

- [基础测试观点：并行加速的计算并不能覆盖访存消耗; 8次浮点加法=1次访存](main.cpp)
- [跨步访问: 缓存行决定数据的粒度](cache.cpp)
> 因为CPU和内存之间隔着缓存，而缓存和内存之间传输数据的最小单位是缓存行（64字节）。16个float是64字节，所以小于64字节的跨步访问，都会导致数据全部被读取出来。而超过64字节的跨步，则中间的缓存行没有被读取，从而变快了。
- [SOA\AOS\AOSOA对比](aos_soa.cpp)
- [预取CPU prefetch 机制 & 延迟隐藏](prefetch.cpp)
> 这一部分的逻辑是这样的：
- 常识：随机访问一定比顺序慢（因为缓存行）
- 但是有些场景如哈希表必须随机访问
- 我们可以按照分随机块的办法，保证局部随机性，不浪费内存带宽
- 但实验发现这样并不能和顺序访问等效
- 是因为CPU的prefetch机制，在计算时顺序把下一个处理的数据取到，随机访问并不能预测出预取位置
- 解决方案：使用更大的随机块，比如4KB，局部线性顺序可以保证prefetch有效
- 最好使用一页4KB大小是因为考虑到OS的内存分页管理，同时为了不跨页，尽可能地对齐，使用`_mm_malloc`手动对齐
- 当然，如果实在要随机访问一小块，可以手动预取`_mm_prefetch`
- 预取视角下的内存瓶颈(mem-bound): 预取访存时间cover住了每个数据计算的时间 (**延迟隐藏**)

- [写入比读取慢2倍](read_write.cpp)
> 写入时缓存行读取了不操作的数据，造成浪费。使用`_mm_stream_si32`直接绕过绕过缓存写入

- 写入1比写入0慢？
> 因为写入0编译器优化为memset, memeset内部使用了stream机制可以更快写入；写入1可以也用stream优化，到达和memset一样的效率

> _mm系列指令出自<xmmintrin.h>头文件。[指令文档](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html)


- 内存分页与池化
    - `malloc` 与 `new int[n]` 不会实际分配内存，用到时再缺页中断; 按页大小分配管理
    - `vector(n)` 可以实际初始化内存，也可以借助帮手类不初始化
    - **重复分配效率低问题**: glibc的malloc实现，不会重复利用现有内存；即使两次分配差不多大的内存，也是会产生却也中断，花费非配时间
    ```cpp
        {
            TICK(first_alloc);
            std::vector<int> arr(n);
            TOCK(first_allcoc)
        }
        {
             TICK(second_alloc);
            std::vector<int> arr(n);
            TOCK(second_allcoc)
        }
        // 两次的时间一样
    ```
    - tbb::cache_aligned_allocator 的最大好处在于他分配的内存地址，永远会对齐到缓存行（64字节），对 SIMD 而言可以用 _mm_load_ps 而不是 _mm_loadu_ps 了。对访存优化而言则意味着可以放心地用 _mm_prefetch，也更高效。
    - 不过其实标准库的 new 和 malloc 已经可以保证 16 字节对齐了。如果你只需要用 _mm_load_ps 而不用 _mm256_load_ps 的话，那直接用标准库的内存分配也没问题。
    > 还有 _mm_malloc(n, aalign) 可以分配对齐到任意 a 字节的内存。他在 <xmmintrin.h> 这个头文件里。是 x86 特有的，并且需要通过 _mm_free 来释放。 还有一个跨平台版本（比如用于 arm 架构）的 aligned_alloc(align, n)，他也可以分配对齐到任意 a 字节的内存，通过 free 释放。 **利用他们可以实现分配对齐到页面（4KB）的内存**。
    - 可以自己实现[AlignedAllocator](alignalloc.h)来指定对齐到任意字节，替代`std::allocator<T>`
    - [手动内存池化，避免每次分配浪费](manual_pool.cpp) 在我的环境上实验效果存疑？


