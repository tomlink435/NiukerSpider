from datetime import datetime

import requests
import json
import time
import re
import pandas as pd
import hashlib

def calculate_hash(content):
    return hashlib.md5(content.strip().encode('utf-8')).hexdigest()

CATEGORY_KEYWORDS = {
    "JUC": [
        # 基本概念
        "线程", "并发", "同步", "锁", "线程安全", "并发编程模型", "线程间通信", "线程池", "多线程", "线程状态","AQS","aqs"

        # 执行器框架
        "Executor", "ExecutorService", "ThreadPoolExecutor", "ScheduledExecutorService", "Executor框架", "线程池",
        "调度池", "线程池管理",
        "线程池调优", "线程池拒绝策略", "线程池参数", "线程池状态", "线程池监控",

        # 同步与互斥
        "同步", "同步方法", "同步块", "锁", "死锁", "重入锁", "悲观锁", "乐观锁", "锁粒度", "锁竞争", "锁优化",
        "锁的性能", "锁降级",
        "ReentrantLock", "ReadWriteLock", "死锁检测", "Condition", "Lock接口", "锁定与解锁",
        "synchronized与ReentrantLock的区别",

        # CountDownLatch & CyclicBarrier
        "CountDownLatch", "CyclicBarrier", "并发同步工具", "计数器", "等待与释放", "同步计数器", "线程协调", "屏障",
        "跨线程协调",

        # Semaphore & BlockingQueue
        "Semaphore", "信号量", "BlockingQueue", "阻塞队列", "生产者消费者", "队列管理", "有界队列", "无界队列",
        "优先级队列", "ArrayBlockingQueue", "LinkedBlockingQueue",

        # Future与异步编程
        "Future", "Callable", "FutureTask", "异步执行", "并发任务", "任务结果", "异步编程模型", "CompletableFuture",
        "并行流", "异步任务", "异步回调",

        # 原子操作与CAS
        "Atomic", "原子类", "AtomicInteger", "AtomicLong", "AtomicReference", "CAS", "Compare and Swap", "无锁编程",
        "原子操作", "并发数据结构",

        # ForkJoin框架
        "ForkJoin", "ForkJoinPool", "分治任务", "并行任务", "工作窃取", "ForkJoinPool调度", "任务分割与合并",
        "ForkJoin任务",

        # ThreadLocal与线程局部变量
        "ThreadLocal", "线程局部变量", "线程隔离", "ThreadLocalMap", "避免共享数据", "ThreadLocal优化", "内存泄漏问题",

        # Phaser与StampedLock
        "Phaser", "阶段性任务", "并发阶段", "并行化任务", "Phaser与CountDownLatch的区别", "StampedLock", "悲观锁",
        "乐观锁", "写锁与读锁", "性能优化",

        # 并发容器
        "并发集合", "ConcurrentHashMap", "CopyOnWriteArrayList", "CopyOnWriteArraySet", "BlockingQueue",
        "ConcurrentLinkedQueue", "ConcurrentSkipListMap", "ConcurrentLinkedDeque", "线程安全集合",

        # 线程池常用类
        "ThreadPoolExecutor", "ScheduledThreadPoolExecutor", "ExecutorService", "ScheduledExecutorService",
        "ForkJoinTask", "任务调度", "线程池核心参数", "线程池大小调优", "线程池拒绝策略",

        # 并发工具类
        "CountDownLatch", "CyclicBarrier", "Semaphore", "Exchanger", "ConcurrentLinkedQueue", "AtomicReference",
        "AtomicInteger", "Phaser", "ThreadLocal", "Executor框架", "Condition",

        # 高并发优化
        "高并发", "高并发编程", "锁优化", "无锁编程", "乐观锁与悲观锁", "读写锁", "锁竞争与锁暴露", "CAS优化",
        "并发瓶颈", "性能调优", "负载均衡", "减少上下文切换",

        # 并发设计模式
        "生产者消费者模式", "单例模式", "线程池设计模式", "观察者模式", "策略模式", "任务拆分与合并", "分治策略",
        "事件驱动模型",

        # 并发调试与监控
        "线程调度", "线程上下文切换", "死锁检测", "线程堆栈分析", "并发调试", "并发监控", "锁竞争监控", "性能分析工具",
        "JVM调优", "线程监控工具", "线程调度器",

        # 并发模型
        "Actor模型", "消息传递模型", "无锁编程模型", "反应式编程", "基于事件的编程模型", "多线程设计模式", "线程池模型",
        "协程模型", "事件驱动模型"
    ],
    "JVM": [
        # 基本概念
        "jvm","JVM", "垃圾回收", "内存管理", "GC", "堆", "栈", "类加载", "内存溢出",

        # 垃圾回收相关
        "垃圾回收算法", "GC算法", "标记清除", "复制算法", "标记整理", "分代收集",
        "年轻代", "老年代", "大对象区", "垃圾回收器", "CMS", "G1垃圾回收器", "Parallel GC", "ZGC", "Shenandoah",
        "回收策略", "Stop-the-world", "内存回收", "垃圾回收日志", "Full GC", "Minor GC", "老年代GC", "GC暂停",

        # 类加载
        "类加载器", "双亲委派模型", "类加载机制", "ClassLoader", "类加载顺序", "类加载过程", "自定义类加载器",

        # JIT编译与优化
        "JIT编译", "即时编译", "热点代码", "逃逸分析", "代码优化", "内联", "方法内联", "热点方法", "编译阈值",
        "C1编译器", "C2编译器",
        "GC与JIT交互", "动态编译", "JVM调优",

        # 内存模型与管理
        "直接内存", "内存映射", "DirectBuffer", "NativeMemory", "堆外内存", "内存屏障", "虚拟内存", "内存映射文件",
        "内存分配", "内存泄漏",

        # 虚拟机栈与线程
        "虚拟机栈", "线程栈", "栈帧", "方法区", "栈帧结构", "栈溢出", "栈的生命周期", "线程管理", "多线程模型",

        # JVM日志和监控
        "GC日志", "JVM日志", "JVM监控", "JVM指标", "JVM性能调优", "内存使用监控", "GC停顿", "JVM Profiler",

        # JVM工具与命令
        "JVM参数", "JVM启动参数", "jconsole", "jvisualvm", "jstack", "jmap", "jstat", "JVM监控工具", "JVM诊断",
        "JVM调优工具",

        # 高级调优与优化
        "JVM调优", "GC调优", "JVM内存优化", "线程优化", "性能瓶颈", "JVM堆优化", "JVM堆外内存管理", "类加载优化",
        "JVM并发优化",

        # 特殊JVM实现
        "HotSpot", "OpenJ9", "GraalVM", "JVM实现", "JVM厂商", "JVM与硬件架构", "JVM平台支持", "JVM与操作系统交互",
        "JVM内存布局",

        # 内存模型与并发
        "Java内存模型",
        # 其他相关
        "逃逸分析", "栈上分配", "锁消除", "内存对齐", "线程局部存储", "JVM规范", "JVM字节码", "字节码优化", "字节码工程师"
    ],
    "Java基础": [
        "继承", "多态", "接口", "抽象类", "类", "对象", "集合", "String",
        "HashMap", "ArrayList", "HashSet", "Comparable", "Comparator", "泛型",
        "反射", "注解", "序列化", "枚举", "流", "IO", "NIO", "Optional",
        "Lambda表达式", "Stream API", "函数式编程", "JDK"
    ],
    "Spring": [
        "Spring", "依赖注入", "Spring Boot", "Spring MVC", "AOP", "事务管理",
        "Bean", "Spring Security", "Spring Cloud", "Spring Data JPA", "Spring Integration",
        "IOC容器", "自动配置", "Starter", "RestTemplate", "WebFlux", "Spring Batch",
        "消息中间件", "RabbitMQ", "Kafka", "Actuator", "Feign", "Eureka", "Zuul", "Gateway"
    ],
    "数据库": [
    # 基础概念
    "SQL", "MySQL", "数据库", "数据库管理系统(DBMS)", "数据库设计", "数据库建模", "关系型数据库", "非关系型数据库", "数据表", "数据行", "数据列",
    "字段", "数据类型", "索引", "主键", "外键", "唯一约束", "检查约束", "默认值", "视图", "触发器", "存储过程", "函数", "查询优化", "性能调优", "数据完整性", "数据安全",

    # 事务管理
    "事务", "ACID", "事务隔离级别", "原子性", "一致性", "隔离性", "持久性", "隔离级别：READ_UNCOMMITTED", "隔离级别：READ_COMMITTED", "隔离级别：REPEATABLE_READ", "隔离级别：SERIALIZABLE", "事务日志", "分布式事务", "两段锁协议", "锁机制",

    # 查询优化
    "查询优化", "执行计划", "查询缓存", "索引优化", "表连接", "视图查询", "嵌套查询", "子查询", "联合查询", "查询重写", "索引扫描", "全表扫描", "查询性能", "慢查询分析", "SQL分析工具",

    # 索引与数据存储
    "索引", "聚集索引", "非聚集索引", "哈希索引", "B+树", "B树", "全文索引", "复合索引", "索引类型", "索引选择性", "索引优化", "索引设计", "索引管理", "索引的选择性", "索引覆盖", "索引的应用场景", "索引的性能影响",

    # 数据库范式
    "范式", "第一范式", "第二范式", "第三范式", "BCNF", "第四范式", "第五范式", "反范式", "冗余", "数据规范化", "反规范化", "数据冗余优化", "范式设计",

    # 高级功能
    "分区表", "分区表设计", "水平分区", "垂直分区", "分区键", "分库分表", "分库策略", "分表策略", "主从复制", "主从架构", "双主复制", "同步复制", "异步复制", "复制延迟", "复制冲突", "复制监控", "双写问题", "一致性",

    # 日志与恢复
    "日志", "Redo Log", "Undo Log", "二进制日志", "事务日志", "回滚", "崩溃恢复", "日志压缩", "日志文件管理", "增量备份", "全量备份", "恢复策略", "日志备份",

    # 数据一致性与并发控制
    "一致性", "强一致性", "最终一致性", "分布式一致性", "CAP理论", "Paxos算法", "Raft算法", "乐观锁", "悲观锁", "锁机制", "MVCC", "并发控制", "锁表", "行级锁", "表级锁", "死锁", "锁竞争", "锁粒度",

    # 数据库备份与恢复
    "备份", "增量备份", "全量备份", "热备份", "冷备份", "数据库恢复", "灾备", "备份策略", "备份工具", "恢复策略", "备份方案", "备份验证", "备份文件加密",

    # 数据库高可用与扩展
    "高可用", "主从复制", "读写分离", "负载均衡", "分布式数据库", "水平扩展", "垂直扩展", "数据库集群", "集群管理", "分布式事务", "CAP理论", "一致性哈希", "分布式锁", "Zookeeper", "Sharding", "ShardingSphere", "TIDB", "MyCat", "数据库中间件", "数据库负载均衡",

    # NoSQL数据库
    "NoSQL", "Redis", "MongoDB", "Cassandra", "HBase", "Elasticsearch", "CouchDB", "Neo4j", "GraphDB", "Key-Value数据库", "文档数据库", "列族数据库", "图数据库", "分布式存储", "大数据存储", "数据模型", "CAP理论", "最终一致性", "ACID与BASE",

    # 数据库安全与加密
    "数据库安全", "数据加密", "数据脱敏", "访问控制", "权限管理", "SQL注入", "防火墙", "防火墙规则", "SQL防注入", "敏感数据管理", "审计日志", "访问审计", "数据泄露", "备份加密", "加密算法", "身份认证", "数据隔离", "隐私保护", "合规性要求"
],
    "Redis": [
        "Redis", "缓存", "内存数据库", "键值对", "数据结构", "过期策略", "LRU",
        "持久化", "RDB", "AOF", "哨兵模式", "主从复制", "分布式锁", "缓存穿透",
        "缓存雪崩", "缓存击穿", "Pub/Sub", "Sorted Set", "Hash", "List",
        "HyperLogLog", "Bitmap", "Stream", "Cluster", "Redis事务", "乐观锁",
        "管道操作", "慢查询日志", "Redis监控", "内存淘汰策略", "Redis优化","Lua脚本"
    ],
    "分布式系统": [
        "分布式", "微服务", "负载均衡", "高可用", "一致性", "CAP理论",
        "Paxos", "Raft", "Zookeeper", "服务注册", "服务发现", "分布式事务",
        "分布式锁", "分布式缓存", "分布式日志", "幂等性", "一致性哈希",
        "消息队列", "Kafka", "RabbitMQ", "ActiveMQ", "RocketMQ", "数据分片"
    ],
    "操作系统": [
        "操作系统", "进程", "线程", "线程池", "内存管理", "文件系统",
        "IO模型", "CPU调度", "死锁", "信号量", "分页", "分段", "虚拟内存",
        "Linux", "Unix", "Shell", "文件权限", "系统调用", "网络编程",
        "Socket", "TCP/IP协议", "UDP协议", "多线程编程"
    ],
    "网络": [
        "网络", "HTTP", "HTTPS", "TCP", "UDP", "IP", "WebSocket", "SSL",
        "TLS", "DNS", "CDN", "代理服务器", "负载均衡", "三次握手", "四次挥手",
        "RESTful", "RPC", "GraphQL", "网络安全", "加密", "数字签名",
        "防火墙", "端口", "网络协议", "抓包分析", "流量控制", "拥塞控制"
    ],
    # 你可以根据需要扩展更多类别和关键词
    "手撕算法": [
        # 基础算法
        "排序", "冒泡排序", "快速排序", "归并排序", "堆排序", "二分查找",
        "双指针", "滑动窗口", "递归", "分治", "回溯", "动态规划", "贪心", "递归与迭代",

        # 数据结构相关
        "链表", "反转链表", "合并链表", "环形链表", "数组", "矩阵", "栈",
        "队列", "优先队列", "双端队列", "哈希表", "二叉树", "二叉搜索树",
        "完全二叉树", "堆", "Trie树", "字典树", "红黑树", "AVL树", "并查集",
        "图", "邻接矩阵", "邻接表", "最短路径", "最小生成树",

        # 字符串处理
        "字符串", "回文串", "最长回文子串", "字符串匹配", "KMP算法", "字典序",
        "正则匹配", "异位词", "最长公共子串", "最长公共子序列", "编辑距离",

        # 动态规划专题
        "动态规划", "背包问题", "01背包", "完全背包", "多重背包",
        "子集和问题", "最长递增子序列", "最长公共子序列", "股票买卖问题",
        "跳跃游戏", "路径问题", "最大子数组和", "划分子集",

        # 图论专题
        "图", "拓扑排序", "DFS", "BFS", "最短路径", "Dijkstra",
        "Floyd-Warshall", "Bellman-Ford", "最小生成树", "Kruskal", "Prim",
        "二分图", "强连通分量", "Tarjan算法", "欧拉回路", "哈密顿回路",

        # 数学算法
        "数学", "质数", "素数筛", "最大公约数", "最小公倍数", "快速幂",
        "大数运算", "排列组合", "斐波那契数列", "约瑟夫环", "素因数分解",

        # 搜索问题
        "搜索", "深度优先搜索", "广度优先搜索", "回溯", "剪枝", "排列问题",
        "组合问题", "全排列", "子集", "迷宫问题", "N皇后",

        # 贪心算法
        "贪心", "区间问题", "活动选择", "最优装载", "霍夫曼编码",
        "最小跳跃次数", "优先队列",

        # 其他常见题型
        "并发问题", "LRU缓存", "LFU缓存", "滑动窗口最大值",
        "区间合并", "区间调度", "矩阵路径", "岛屿问题",
        "水塘抽样", "随机算法", "洗牌算法", "位运算", "异或运算"
    ],
}


def categorize_content(content):
    """
    根据面经内容进行分类，返回分类结果。
    :param content: 面经内容
    :return: 分类标签（如 JUC, JVM, Java基础）
    """
    matched_categories = []
    for category, keywords in CATEGORY_KEYWORDS.items():
        for keyword in keywords:
            if keyword in content:
                matched_categories.append(category)
    return matched_categories if matched_categories else ["其他"]  # 如果没有匹配到任何分类，返回 "其他"


def _parse_newcoder_page(data):
    res = []
    for x in data['data']['records']:
        dic = {}
        dic['user'] = x['userBrief']['nickname']
        dic['contentId'] = x['contentId']
        x = x['contentData'] if 'contentData' in x else x['momentData']
        dic['title'] = x['title']
        dic['content'] = x['content']
        dic['time'] = edit_time(x['editTime'])

        # 拆分并分类内容
        content_parts = x['content'].split("\n")  # 按换行符分段
        categorized_parts = []

        for part in content_parts:
            categories = categorize_content(part)
            categorized_parts.append({"content": part, "categories": categories})

        dic['categorized_content'] = categorized_parts
        res.append(dic)
    return res
def edit_time(time):
    return  datetime.fromtimestamp(time/1000)

def get_newcoder_page(page=1):
    header = {
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.0.0 Safari/537.36",
        "content-type": "application/json"
    }

    payload = {"companyList": [], "jobId": 11002, "level": 3, "order": 3, "page": page, "isNewJob": True}

    x = requests.post('https://gw-c.nowcoder.com/api/sparta/job-experience/experience/job/list?_=1735811139897',
                      json=payload, headers=header)
    data = _parse_newcoder_page(x.json())

    return data


def run():
    res = []
    for i in range(1, 3000):  # 你可以根据需要调整页数
        try:
            print(f"Fetching page {i}...")
            page = get_newcoder_page(i)
            if not page:
                break
            res.extend(page)
            time.sleep(0.5)  # 防止请求过快，模拟正常访问
        except Exception as e:
            print(f"Error on page {i}: {e}")
            save_results_to_excel(res, filename="interview_experience_partial.xlsx")
            raise e
    return res


def save_results_to_excel(data, filename="interview_experience_by_category.xlsx"):
    # 将数据保存为Excel文件
    all_records = []
    unique_records = set()
    for record in data:
        for part in record['categorized_content']:
            content_hash = calculate_hash(part['content'])
            if content_hash in unique_records:  # 如果哈希值已存在，则跳过
                continue
            unique_records.add(content_hash)
            all_records.append({
                'user': record['user'],
                'contentId': record['contentId'],
                'title': record['title'],
                'content': part['content'],
                'categories': ', '.join(part['categories']),
                'time': record['time']
            })

    df = pd.DataFrame(all_records)  # 将拆分后的记录转换为DataFrame
    df.to_excel(filename, index=False, engine='openpyxl')  # 保存为Excel文件
    print(f"Results saved to {filename}")


def main():
    # 获取面试经验数据
    interview_data = run()

    # 保存数据到Excel文件
    save_results_to_excel(interview_data)


if __name__ == "__main__":
    main()