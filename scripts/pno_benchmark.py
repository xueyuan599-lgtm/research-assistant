# from collections import deque
# enque=deque()
# enque.append("底部")
# enque.append("中部")
# enque.append("顶部")
# print(enque)

stack=[]
stack.append("底部的书")
stack.append("中间的书")
stack.append("顶部的书")
print("当前栈:", stack)
top_book=stack.pop()
print(top_book)
print(stack)