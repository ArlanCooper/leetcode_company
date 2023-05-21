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

## 面试题 02.05. 链表求和
### 题目描述
给定两个用链表表示的整数，每个节点包含一个数位。

这些数位是反向存放的，也就是个位排在链表首部。

编写函数对这两个整数求和，并用链表形式返回结果。

### 示例
```
输入：(7 -> 1 -> 6) + (5 -> 9 -> 2)，即617 + 295
输出：2 -> 1 -> 9，即912
```


### 解法
```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        dummy = ListNode(-1)
        cur = dummy
        carry = 0
        while l1 or l2 or carry:
            t = carry
            if l1:
                t += l1.val
                l1 = l1.next 
            if l2:
                t += l2.val 
                l2 = l2.next
            carry = t // 10
            cur.next = ListNode(t % 10)
            cur = cur.next

        return dummy.next

```

## 470. 用 Rand7() 实现 Rand10()
### 题目描述
给定方法 rand7 可生成 [1,7] 范围内的均匀随机整数，试写一个方法 rand10 生成 [1,10] 范围内的均匀随机整数。
你只能调用 rand7() 且不能调用其他方法。请不要使用系统的 Math.random() 方法。
每个测试用例将有一个内部参数 n，即你实现的函数 rand10() 在测试时将被调用的次数。请注意，这不是传递给 rand10() 的参数。
### 示例
```
输入: 1
输出: [2]

```

### 解题思路
我们可以用拒绝采样的方法实现 Rand10()。在拒绝采样中，如果生成的随机数满足要求，那么就返回该随机数，否则会不断生成，直到生成一个满足要求的随机数为止。
我们只需要能够满足等概率的生成 10个不同的数即可，具体的生成方法可以有很多种，比如我们可以利用两个 Rand7()相乘，我们只取其中等概率的10个不同的数的组合即可，当然还有许多其他不同的解法，可以利用各种运算和函数的组合等方式来实现。

### 解法
```python
# The rand7() API is already defined for you.
# def rand7():
# @return a random integer in the range 1 to 7

class Solution:
    def rand10(self) -> int:
        while True:
            row = rand7()
            col = rand7()
            idx = (row - 1) * 7 + col
            if idx <= 40:
                return 1 + (idx - 1) % 10


```

## 179. 最大数
### 题目描述
给定一组非负整数 nums，重新排列每个数的顺序（每个数不可拆分）使之组成一个最大的整数。
注意：输出结果可能非常大，所以你需要返回一个字符串而不是整数。


### 示例
```
输入：nums = [10,2]
输出："210"
```
```
输入：nums = [3,30,34,5,9]
输出："9534330"
```

### 解题思路
如果拼接得到的字符串结果更大的话，那么原本的整型的数字拼接结果也一定更大吗？比如 "210" > "102"，那么一定能得到 210 > 102 么？

答案是肯定的：首先拼接成的两个字符串一定是等长的。等长的字符串在比较的时候，是按照字符串的各个字符从前向后逐个比较的，所以相当于先比较了百分位，然后比较十分位，最后比较个位。所以在字符串等长的情况下，字符串大，那么对应的整型也更大。但两个不等长的字符串就没有这个结论了， 比如 "2" > "10"，但是 2 < 10。

综上，我们按照下面的步骤：
先把 nums 中的所有数字转字符串，形成字符串数组 nums_str；
比较两个字符串 x,y 拼接的结果 x + y 和 y + x 哪个更大，从而确定 x 和 y 谁排在前面；将 nums_str 降序排序；
把整个数组排序的结果拼接成一个字符串，并返回。

### 解法
```python
from typing import List
import functools
class Solution:
    #先把nums中的所有数字转化为字符串，形成字符串数组 nums_str
    #比较两个字符串x,y的拼接结果x+y和y+x哪个更大，从而确定x和y谁排在前面；将nums_str降序排序
    #把整个数组排序的结果拼接成一个字符串，并且返回
    def largestNumber(self, nums: List[int]) -> str:
        nums_str=list(map(str,nums))
        compare=lambda x,y: 1 if x+y<y+x else -1
        nums_str.sort(key=functools.cmp_to_key(compare))
        res=''.join(nums_str)
        if res[0]=='0':
            res='0'
        return res
```


## 309. 最佳买卖股票时机含冷冻期
### 题目描述
给定一个整数数组prices，其中第  prices[i] 表示第 i 天的股票价格 。
设计一个算法计算出最大利润。在满足以下约束条件下，你可以尽可能地完成更多的交易（多次买卖一支股票）:
卖出股票后，你无法在第二天买入股票 (即冷冻期为 1 天)。
注意：你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。

### 示例
```
输入: prices = [1,2,3,0,2]
输出: 3 
解释: 对应的交易状态为: [买入, 卖出, 冷冻期, 买入, 卖出]
```

```
输入: prices = [1]
输出: 0
```

### 解题思路
我们用 f[i]表示第 i 天结束之后的「累计最大收益」。根据题目描述，由于我们最多只能同时买入（持有）一支股票，并且卖出股票后有冷冻期的限制，因此我们会有三种不同的状态：

我们目前持有一支股票，对应的「累计最大收益」记为 f[i][0]；

我们目前不持有任何股票，并且处于冷冻期中，对应的「累计最大收益」记为 f[i][1]；

我们目前不持有任何股票，并且不处于冷冻期中，对应的「累计最大收益」记为 f[i][2]。

这里的「处于冷冻期」指的是在第 iii 天结束之后的状态。也就是说：如果第 iii 天结束之后处于冷冻期，那么第 i+1天无法买入股票。

如何进行状态转移呢？在第 iii 天时，我们可以在不违反规则的前提下进行「买入」或者「卖出」操作，此时第 iii 天的状态会从第 i−1天的状态转移而来；我们也可以不进行任何操作，此时第 i 天的状态就等同于第 i−1天的状态。那么我们分别对这三种状态进行分析：

对于 f[i][0]，我们目前持有的这一支股票可以是在第 i−1天就已经持有的，对应的状态为f[i−1][0]；或者是第 i 天买入的，那么第 i−1天就不能持有股票并且不处于冷冻期中，对应的状态为 f[i−1][2]加上买入股票的负收益 prices[i]。因此状态转移方程为：f[i][0]=max⁡(f[i−1][0],f[i−1][2]−prices[i])
对于 f[i][1]，我们在第 i天结束之后处于冷冻期的原因是在当天卖出了股票，那么说明在第i−1天时我们必须持有一支股票，对应的状态为f[i−1][0]加上卖出股票的正收益 prices[i]。因此状态转移方程为：
f[i][1]=f[i−1][0]+prices[i]
对于 f[i][2]，我们在第 iii 天结束之后不持有任何股票并且不处于冷冻期，说明当天没有进行任何操作，即第i−1天时不持有任何股票：如果处于冷冻期，对应的状态为 f[i−1][1]；如果不处于冷冻期，对应的状态为f[i−1][2]。因此状态转移方程为：f[i][2]=max⁡(f[i−1][1],f[i−1][2])
这样我们就得到了所有的状态转移方程。如果一共有 nnn 天，那么最终的答案即为：max⁡(f[n−1][0],f[n−1][1],f[n−1][2])
注意到如果在最后一天（第 n−1n-1n−1 天）结束之后，手上仍然持有股票，那么显然是没有任何意义的。因此更加精确地，最终的答案实际上是 f[n−1][1]和 f[n−1][2]中的较大值，即：max⁡(f[n−1][1],f[n−1][2])

### 解法
```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        if not prices:
            return 0
        
        n = len(prices)
        # f[i][0]: 手上持有股票的最大收益
        # f[i][1]: 手上不持有股票，并且处于冷冻期中的累计最大收益
        # f[i][2]: 手上不持有股票，并且不在冷冻期中的累计最大收益
        f = [[-prices[0], 0, 0]] + [[0] * 3 for _ in range(n - 1)]
        for i in range(1, n):
            f[i][0] = max(f[i - 1][0], f[i - 1][2] - prices[i])
            f[i][1] = f[i - 1][0] + prices[i]
            f[i][2] = max(f[i - 1][1], f[i - 1][2])
        
        return max(f[n - 1][1], f[n - 1][2])

```

## 122. 买卖股票的最佳时机 II
### 题目描述
给定一个整数数组prices，其中第  prices[i] 表示第 i 天的股票价格 。
设计一个算法计算出最大利润。在满足以下约束条件下，你可以尽可能地完成更多的交易（多次买卖一支股票）:
卖出股票后，你无法在第二天买入股票 (即冷冻期为 1 天)。
注意：你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。

### 示例
```
输入：prices = [7,1,5,3,6,4]
输出：7
解释：在第 2 天（股票价格 = 1）的时候买入，在第 3 天（股票价格 = 5）的时候卖出, 这笔交易所能获得利润 = 5 - 1 = 4 。
     随后，在第 4 天（股票价格 = 3）的时候买入，在第 5 天（股票价格 = 6）的时候卖出, 这笔交易所能获得利润 = 6 - 3 = 3 。
     总利润为 4 + 3 = 7 。

```

### 解题思路
考虑到「不能同时参与多笔交易」，因此每天交易结束后只可能存在手里有一支股票或者没有股票的状态。
定义状态 dp[i][0]表示第i天交易完后手里没有股票的最大利润，dp[i][1]表示第i天交易完后手里持有一支股票的最大利润（i从 0开始）。
考虑 dp[i][0]的转移方程，如果这一天交易完后手里没有股票，那么可能的转移状态为前一天已经没有股票，即 dp[i−1][0]，或者前一天结束的时候手里持有一支股票，即 dp[i−1][1]，这时候我们要将其卖出，并获得 prices[i]的收益。因此为了收益最大化，我们列出如下的转移方程：dp[i][0]=max⁡{dp[i−1][0],dp[i−1][1]+prices[i]}
再来考虑 dp[i][1]，按照同样的方式考虑转移状态，那么可能的转移状态为前一天已经持有一支股票，即 dp[i−1][1]，或者前一天结束时还没有股票，即 dp[i−1][0]，这时候我们要将其买入，并减少 prices[i]的收益。可以列出如下的转移方程：dp[i][1]=max⁡{dp[i−1][1],dp[i−1][0]−prices[i]}
对于初始状态，根据状态定义我们可以知道第 000 天交易结束的时候 dp[0][0]=0。
因此，我们只要从前往后依次计算状态即可。由于全部交易结束后，持有股票的收益一定低于不持有股票的收益，因此这时候 dp[n−1][0]的收益必然是大于 dp[n−1][1]的，最后的答案即为 dp[n−1][0]。

### 解法
```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        n = len(prices)
        dp = [[0,-prices[0]]] + [[0,0] for i in range(n-1)]
        for i in range(1,n):
            dp[i][0] = max(dp[i-1][0], dp[i-1][1] + prices[i])
            dp[i][1] = max(dp[i-1][1],dp[i-1][0] - prices[i])
        return dp[n-1][0]
```

```python
## 优化空间
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        n = len(prices)
        dp0,dp1 = 0, -prices[0]
        for i in range(1,n):
            new_dp0 = max(dp0,dp1 + prices[i])
            new_dp1 = max(dp1, dp0 - prices[i])
            dp0 = new_dp0
            dp1 = new_dp1
        return dp0

```
