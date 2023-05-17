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


## 152. 乘积最大子数组
### 题目描述
给你一个整数数组 nums ，请你找出数组中乘积最大的非空连续子数组（该子数组中至少包含一个数字），并返回该子数组所对应的乘积。
测试用例的答案是一个 32-位 整数。
子数组 是数组的连续子序列。

### 示例
```
输入: nums = [2,3,-2,4]
输出: 6
解释: 子数组 [2,3] 有最大乘积 6。

```

```
输入: nums = [-2,0,-1]
输出: 0
解释: 结果不能为 2, 因为 [-2,-1] 不是子数组。

```

### 解法
```python

class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        if not nums:
            return
        pre_min = nums[0]
        pre_max = nums[0]
        res = nums[0]
        for i in range(1,len(nums)):
            now_max = max(pre_max*nums[i],pre_min*nums[i],nums[i])
            now_min = min(pre_min * nums[i],pre_max*nums[i],nums[i])
            res = max(res,now_max)
            pre_max = now_max
            pre_min = now_min
        return res


```

## 236. 二叉树的最近公共祖先

### 题目描述
给定一个二叉树, 找到该树中两个指定节点的最近公共祖先。
百度百科中最近公共祖先的定义为：“对于有根树 T 的两个节点 p、q，最近公共祖先表示为一个节点 x，满足 x 是 p、q 的祖先且 x 的深度尽可能大（一个节点也可以是它自己的祖先）。”

### 示例
```
输入：root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 1
输出：3
解释：节点 5 和节点 1 的最近公共祖先是节点 3 。

```


```
输入：root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 4
输出：5
解释：节点 5 和节点 4 的最近公共祖先是节点 5 。因为根据定义最近公共祖先节点可以为节点本身。
```


### 解法
```python

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def lowestCommonAncestor(self, root: TreeNode, p: TreeNode, q: TreeNode) -> TreeNode:
        if not root or root == p or root == q: return root
        left = self.lowestCommonAncestor(root.left, p, q)
        right = self.lowestCommonAncestor(root.right, p, q)
        if not left: return right
        if not right: return left
        return root
```

## 1124. 表现良好的最长时间段
### 题目描述
给你一份工作时间表 hours，上面记录着某一位员工每天的工作小时数。
我们认为当员工一天中的工作小时数大于 8 小时的时候，那么这一天就是「劳累的一天」。
所谓「表现良好的时间段」，意味在这段时间内，「劳累的天数」是严格 大于「不劳累的天数」。
请你返回「表现良好时间段」的最大长度。

### 示例
```
输入：hours = [9,9,6,0,6,6,9]
输出：3
解释：最长的表现良好时间段是 [9,9,6]。
```

### 解题思路
前缀和+单调栈


### 解法
```python

class Solution:
    def longestWPI(self, hours: List[int]) -> int:
        n = len(hours)
        s = [0] * (n + 1)  # 前缀和
        st = [0]  # s[0]
        for j, h in enumerate(hours, 1):
            s[j] = s[j - 1] + (1 if h > 8 else -1)
            if s[j] < s[st[-1]]: st.append(j)  # 感兴趣的 j
        ans = 0
        for i in range(n, 0, -1):
            while st and s[i] > s[st[-1]]:
                ans = max(ans, i - st.pop())  # [st[-1],i) 可能是最长子数组
        return ans

```

