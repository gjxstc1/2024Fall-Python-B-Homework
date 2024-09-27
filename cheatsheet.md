## 1 保留几位小数（无import）
```python
ans = 0
for i in range(n):
    if w >= a[i].weight:
        w -= a[i].weight
        ans += a[i].value
    else:
        ans += a[i].value * w / a[i].weight
        break
print(f"{ans:.1f}")
```

-------

## 2 持续输入
```python
while True:
  try:
    输入
  except EOFError:
    break
```

------

## 3
