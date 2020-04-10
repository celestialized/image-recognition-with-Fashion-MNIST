# Broadcast testing
To see if we may perform element wise operations on tensors we need to figure out if they are compatible and then determine the resulting size if they are, and finally perform the appropriate element wise operation/action on them. Broadcasting goal is to get them to the same shape to operate on them together. If there is no way to reshape them to be compatible you may not be able to do as you wish. See [this](https://deeplizard.com/learn/video/6_33ulFDuCg) for more info on broadcasting.

## Are they compatible?
We line them up starting from their last dimensions. They are compatible if EITHER of the following are true: The numbers are equal, or one of them is a one.
```py
# compare
# (3,1)
# (1,3)
# ----- result works from right to left, and both pass the test above
# (3,3) is the shape needed for an operation with max.
```
Now figure out how we got the shape needed. Take the max for each starting again with the last dimension working back. If you have two with different shapes, substitute a one for the blank spaces.
```py
# test
# (1,2,3)
# ( ,3,1)
# -------

# becomes
# (1,2,3)
# (1,3,1)
# ------- tests all do not pass, result as follows with max on each column
# (x,x,3) stop on your second test 2 != 3. Done.
```