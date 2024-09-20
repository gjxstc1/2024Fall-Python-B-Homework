# Assignment #1: 自主学习

Updated 0110 GMT+8 Sep 10, 2024

2024 fall, Complied by 高景行 数学科学学院



**说明：**

1）请把每个题目解题思路（可选），源码Python, 或者C++（已经在Codeforces/Openjudge上AC），截图（包含Accepted），填写到下面作业模版中（推荐使用 typora https://typoraio.cn ，或者用word）。AC 或者没有AC，都请标上每个题目大致花费时间。

3）课程网站是Canvas平台, https://pku.instructure.com, 学校通知9月19日导入选课名单后启用。**作业写好后，保留在自己手中，待9月20日提交。**

提交时候先提交pdf文件，再把md或者doc文件上传到右侧“作业评论”。Canvas需要有同学清晰头像、提交文件有pdf、"作业评论"区有上传的md或者doc附件。

4）如果不能在截止前提交作业，请写明原因。



## 1. 题目

### 02733: 判断闰年

http://cs101.openjudge.cn/practice/02733/



思路：
直接判断


##### 代码

```python
# 
n = int(input())
if n % 4 or (n % 100 == 0 and n % 400):
    print("N")
else:
    print("Y")
```



代码运行截图 ==（至少包含有"Accepted"）==
![alt text](WeChat58689d65be02ca764aae52a32cc106ba.jpg)



### 02750: 鸡兔同笼

http://cs101.openjudge.cn/practice/02750/



思路：
显然$n$要是偶数。最多动物时全是鸡；最少动物时让兔子尽量多，$mod$ $4$讨论


##### 代码

```python
# 
n = int(input())
if n % 2:
    print("0 0")
else:
    a = n // 2
    if n % 4:
        b = (n - 2) // 4 + 1
    else:
        b = n // 4
    print(b, end = " ")
    print(a)

```



代码运行截图 ==（至少包含有"Accepted"）==
![alt text](WeChat07013aa174f6062e03d6830973f3e6d7.jpg)




### 50A. Domino piling

greedy, math, 800, http://codeforces.com/problemset/problem/50/A



思路：
$m,n$中有偶数时按偶数方向放，全能放下；$m,n$全为奇数时一定至少空余一格放不了，而让最后一行空出边上一格后剩余全可放下、前$m - 1$行用前一种情况知全可放下，故可只空一格不放。

##### 代码

```python
# 
m, n = map(int, input().split(" "))
if m % 2 == 0 or n % 2 == 0:
    print(m * n // 2)
else:
    print((m * n - 1) // 2)
```

代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![alt text](WeChatec92e864c82aecdc6d858e7b46214fd7-1.jpg)





### 1A. Theatre Square

math, 1000, https://codeforces.com/problemset/problem/1/A



思路：
容易知道贪心覆盖即可（证明时可考虑所有横纵坐标$mod$ $a$为1的格子，这些格子只能被不同的$a * a$覆盖，故可证明至少$\lceil\frac{n}{a}\rceil*\lceil\frac{m}{a}\rceil$个，这与贪心的构造相同）。再用$\lceil\frac{p}{q}\rceil = [\frac{p - 1}{q}] + 1$即可。


##### 代码

```python
# 
n, m, a = map(int, input().split())
x = (n - 1) // a + 1
y = (m - 1) // a + 1
print(x * y)

```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![alt text](WeChat1ff54feb90839a560c0fdd0fcd0f70fe.jpg)



### 112A. Petya and Strings

implementation, strings, 1000, http://codeforces.com/problemset/problem/112/A



思路：
全转成小写后再做即可


##### 代码

```python
# 
a = input()
b = input()
a = a.lower()
b = b.lower()
n = len(a)
for i in range(n):
    if a[i] < b[i]:
        print(-1)
        exit()
    if a[i] > b[i]:
        print(1)
        exit()
print(0)
```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==
![alt text](WeChat8747ca768110e945d86c75c6ab940454.jpg)




### 231A. Team

bruteforce, greedy, 800, http://codeforces.com/problemset/problem/231/A



思路：
直接做。


##### 代码

```python
# 
n = int(input())
cnt = 0
for i in range(n):
    x, y, z = map(int, input().split(" "))
    if x + y + z >= 2:
        cnt += 1

print(cnt)
```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![alt text](WeChatec3ac6a78691d15dc95970ffc1a6e6ff.jpg)



## 2. 学习总结和收获

==如果作业题目简单，有否额外练习题目，比如：OJ“计概2024fall每日选做”、CF、LeetCode、洛谷等网站题目。==

我学习了python的基本语法，认为主要难点在于python的读入方式以及涉及到字符串的读入、换大小写、转换成其他数据类型等。

OJ“计概2024fall每日选做”的题目目前为止全部做完并全部AC。
