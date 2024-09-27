# Assignment #2: 语法练习

Updated 0126 GMT+8 Sep 24, 2024

2024 fall, Complied by ==高景行 数学科学学院==



**说明：**

1）请把每个题目解题思路（可选），源码Python, 或者C++（已经在Codeforces/Openjudge上AC），截图（包含Accepted），填写到下面作业模版中（推荐使用 typora https://typoraio.cn ，或者用word）。AC 或者没有AC，都请标上每个题目大致花费时间。

3）课程网站是Canvas平台, https://pku.instructure.com, 学校通知9月19日导入选课名单后启用。**作业写好后，保留在自己手中，待9月20日提交。**

提交时候先提交pdf文件，再把md或者doc文件上传到右侧“作业评论”。Canvas需要有同学清晰头像、提交文件有pdf、"作业评论"区有上传的md或者doc附件。

4）如果不能在截止前提交作业，请写明原因。



## 1. 题目

### 263A. Beautiful Matrix

https://codeforces.com/problemset/problem/263/A



思路：
显然次数为1的位置到最中心点的$Hamilton$距离


##### 代码

```python
# 
a = []
for __ in range(5):
    a.append(list(map(int, input().split())))
for i in range(5):
    for j in range(5):
        if a[i][j] == 1:
            print(abs(2 - i) + abs(2 - j))
            exit()
```



代码运行截图 ==（至少包含有"Accepted"）==

![alt text](WeChat3ffb121760b52f335130680a059de9f2-1.jpg)



### 1328A. Divisibility Problem

https://codeforces.com/problemset/problem/1328/A



思路：
简单同余即可。


##### 代码

```python
# 
T = int(input())
for __ in range(T):
    a, b = map(int, input().split())
    if a % b == 0: print(0)
    else: print(b - a % b)

```



代码运行截图 ==（至少包含有"Accepted"）==
![alt text](WeChataab29e707cd9bbfd067058a84223abc7-1.jpg)




### 427A. Police Recruits

https://codeforces.com/problemset/problem/427/A



思路：
按时间戳遍历：如果歹徒更多，那么就放走歹徒，注意要清零歹徒数量。


##### 代码

```python
# 
n = int(input())
a = list(map(int, input().split()))
sum = 0
ans = 0
for i in range(n):
    sum += a[i]
    if sum < 0:
        ans += 1
        sum = 0
print(ans)

```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==
![alt text](WeChat5f78f0631747a5d370296f6785f00109-2.jpg)



### 02808: 校门外的树

http://cs101.openjudge.cn/practice/02808/



思路：

这个做法明显优于一般模拟，是$O(\max(n,m))$做法（可做$n, m\approx1e7$）。

把种树的区间想象为左右括号，在两端做标记区分左右括号（如用$+-1$）。一个位置种上树当且仅当其前面左括号数$>$右括号数(因树一定在某个配对的左右括号内)，故只用遍历一次计数即可。


##### 代码

```python
# 
n, m = map(int, input().split(" "))
x = [False]
count = [0] * (n + 2)
for www in range(m):
    l, r = map(int, input().split(" "))
    count[l] -= 1
    count[r + 1] += 1
sum = 0
ans = 0
for i in range(n + 1):
    sum += count[i]
    if sum >= 0:
        ans += 1
print(ans)
```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![alt text](WeChatd0598653cb38005bc1b017a30c68b4cf-1.jpg)



### sy60: 水仙花数II

https://sunnywhy.com/sfbj/3/1/60



思路：
暴力预处理所有满足的数


##### 代码

```python
# 
flag = [False for __ in range(1, 1001)]
for i in range(1, 10):
    for j in range(0, 10):
        for k in range(0, 10):
            if i * 100 + j * 10 + k == i ** 3 + j ** 3 + k ** 3:
                flag[i * 100 + j * 10 + k] = True
a, b = map(int, input().split())
now = 0
for i in range(a, b + 1):
    if flag[i]:
        if now == 0: print(i, end = "")
        else:
            print(" ", end = "")
            print(i, end = "")
        now += 1
if now == 0:
    print("NO", end = "")
```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==
![alt text](WeChat947cbf48d7ed16c4798e6dfaff7a7e3f-1.jpg)




### 01922: Ride to School

http://cs101.openjudge.cn/practice/01922/



思路：
这是有深刻思想的模拟，想了非常长时间。关键在于只用考虑最后到达时Charley在跟谁，那么所用时间就是这个人走全长花费时间，这是固定的且容易求的。还有一个观察是不用考虑$t<0$的人，这是因为如果这个人速度快Charley就追不上，如果速度慢被反超了Charley也不会跟着他，因此最后到达时Charley一定不会跟着$t<0$的人（用到最开始时Charley不跟着这种人）。

口胡：如果想不到这种模拟，那么只能将人按照速度排序，然后用搜索看现在在跟着谁，每次找后面有谁会追上当前的人并转移到这个人。注意到这满足贪心，即每次只用往速度更大的人那里找，所以复杂度是$O(n)$，但是可能面临精度被卡等问题（没测试过）。

##### 代码

```python
# 
import cmath
import math
while True:
    n = int(input())
    if n == 0: break
    mn = float("inf")
    for _ in range(n):
        v, t = map(int, input().split())
        if t < 0: continue
        mn = min(mn, math.ceil(4.5 / v * 3600 + t))
    print(mn)
```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![alt text](WeChat4084ea77af8166b2c376d19652b0adda-1.jpg)



## 2. 学习总结和收获

==如果作业题目简单，有否额外练习题目，比如：OJ“计概2024fall每日选做”、CF、LeetCode、洛谷等网站题目。==

我做了一些小模拟问题，这些问题不乏数学思想，但都可以避免用复杂工具、高复杂度程序，而用简单巧妙的办法解决。我也更加熟练掌握语法(如无穷大无穷小)。我收获颇丰。

OJ“计概2024fall每日选做”目前全部AC。
