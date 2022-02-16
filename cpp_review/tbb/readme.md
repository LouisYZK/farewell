# TBB 并行编程
tbb是intel研发的并行编程框架，竞品主要是OpenMP

- parallel_invoke, 并行调用，str字符并行查找[例子](str_para.cpp)
- 并行映射 parallel_map, [map](map.cpp)
- 并行缩并 parallel_reduce, [reduce](reduce.cpp)
- 并行扫描 parallel_scan, [scan](scan.cpp)
- 任务粒度 [grain](grain.cpp)
- [并发容器](concurrent_ctn.cpp)
- [并行筛选](filter.cpp)
- [并行快速排序--优于td::sort](quick_sort.cpp)
- [流水线](pipeline.cpp)