各个大厂刷题
===


# 拼多多

## 题目列表

```
剑指 Offer 41. 数据流中的中位数
829. 连续整数求和
152. 乘积最大子数组
54. 螺旋矩阵
236. 二叉树的最近公共祖先
480. 滑动窗口中位数
622. 设计循环队列
1124. 表现良好的最长时间段
470. 用 Rand7() 实现 Rand10()
391. 完美矩形
面试题 10.05. 稀疏数组搜索
面试题 02.05. 链表求和
706. 设计哈希映射
179. 最大数
268. 丢失的数字
143. 重排链表
61. 旋转链表
309. 最佳买卖股票时机含冷冻期
122. 买卖股票的最佳时机 II
5. 最长回文子串
215. 数组中的第K个最大元素
2. 两数相加
1. 两数之和
3. 无重复字符的最长子串
21. 合并两个有序链表
56. 合并区间
72. 编辑距离
78. 子集
88. 合并两个有序数组
110. 平衡二叉树
139. 单词拆分
153. 寻找旋转排序数组中的最小值
191. 位1的个数
224. 基本计算器
225. 用队列实现栈
235. 二叉搜索树的最近公共祖先
260. 只出现一次的数字 III
316. 去除重复字母
407. 接雨水 II
46. 全排列
287. 寻找重复数
43. 字符串相乘
11. 盛最多水的容器
53. 最大子数组和
149. 直线上最多的点数
94. 二叉树的中序遍历
395. 至少有 K 个重复字符的最长子
19. 删除链表的倒数第 N 个结点
40. 组合总和 II
207. 课程表
```



## 剑指 Offer 41. 数据流中的中位数

### 题目描述
如何得到一个数据流中的中位数？如果从数据流中读出奇数个数值，那么中位数就是所有数值排序之后位于中间的数值。如果从数据流中读出偶数个数值，那么中位数就是所有数值排序之后中间两个数的平均值。

例如，

[2,3,4] 的中位数是 3

[2,3] 的中位数是 (2 + 3) / 2 = 2.5

设计一个支持以下两种操作的数据结构：

void addNum(int num) - 从数据流中添加一个整数到数据结构中。
double findMedian() - 返回目前所有元素的中位数。

### 示例
```
输入：
["MedianFinder","addNum","addNum","findMedian","addNum","findMedian"]
[[],[1],[2],[],[3],[]]
输出：[null,null,null,1.50000,null,2.00000]

```



### 解1：

```python
from heapq import *

class MedianFinder:
    def __init__(self):
        self.A = [] # 小顶堆，保存较大的一半
        self.B = [] # 大顶堆，保存较小的一半

    def addNum(self, num: int) -> None:
        if len(self.A) != len(self.B):
            heappush(self.A, num)
            heappush(self.B, -heappop(self.A))
        else:
            heappush(self.B, -num)
            heappush(self.A, -heappop(self.B))

    def findMedian(self) -> float:
        return self.A[0] if len(self.A) != len(self.B) else (self.A[0] - self.B[0]) / 2.0


```

### 解2:
```python

from heapq import *

class MedianFinder:
    def __init__(self):
        self.A = [] # 小顶堆，保存较大的一半
        self.B = [] # 大顶堆，保存较小的一半

    def addNum(self, num: int) -> None:
        if len(self.A) != len(self.B):
            heappush(self.A, num)
            heappush(self.B, -heappop(self.A))
        else:
            heappush(self.B, -num)
            heappush(self.A, -heappop(self.B))

    def findMedian(self) -> float:
        return self.A[0] if len(self.A) != len(self.B) else (self.A[0] - self.B[0]) / 2.0


```


