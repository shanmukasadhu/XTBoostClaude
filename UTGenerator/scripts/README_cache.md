# XTBoost 本地仓库缓存功能

本功能允许XTBoost在本地缓存SWE-bench数据集中的仓库和commit，避免每次运行时都从GitHub下载。

## 设置方法

1. 首先，创建一个本地缓存目录（默认为项目根目录下的`repo_cache`）：

```bash
mkdir -p repo_cache
```

2. 设置环境变量（可选）：

```bash
export LOCAL_REPO_CACHE="/path/to/your/cache/directory"
```

如果不设置此环境变量，系统将使用默认的`repo_cache`目录。

## 预先下载和缓存仓库

使用提供的脚本预先下载和缓存所有仓库：

```bash
# 缓存SWE-bench Lite数据集中的仓库
python -m UTGenerator.scripts.cache_repos --dataset_split lite

# 缓存SWE-bench Verified数据集中的仓库
python -m UTGenerator.scripts.cache_repos --dataset_split verified

# 指定自定义缓存目录
python -m UTGenerator.scripts.cache_repos --dataset_split lite --cache_dir /path/to/cache
```

## 工作原理

系统会按照以下顺序查找仓库数据：

1. 首先检查`PROJECT_FILE_LOC`环境变量指定的目录中是否有预处理的结构文件
2. 然后检查本地缓存目录中是否有缓存的结构文件
3. 如果以上都没有找到，才会从GitHub下载，并将结果保存到缓存中

缓存目录结构如下：

```
repo_cache/
├── django_django/
│   ├── commit1.json
│   ├── commit2.json
│   └── ...
├── sphinx-doc_sphinx/
│   ├── commit1.json
│   └── ...
└── ...
```

## 注意事项

- 首次运行时，系统仍需从GitHub下载仓库，但之后的运行将使用缓存
- 缓存文件可能会占用较大的磁盘空间，请确保有足够的存储空间
- 如果GitHub仓库更新，缓存不会自动更新，需要手动删除缓存文件或目录 