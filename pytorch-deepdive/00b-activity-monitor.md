# Activity Monitor
Mac tool may come in helpful for looking a processes here. As we start digging into things like disabling and enabling derivative capturing: 

```py
In [8]: torch.set_grad_enabled(True)                
Out[8]: <torch.autograd.grad_mode.set_grad_enabled at 0x11bc22630>
```

Command + Shift and enter "activity monitor" to quick launch.

* `ps ax | grep vscode` on mac spits out "gibberish", we want PID and GPID quick
  * don't trust the container in the container?
  * want a kill all zombies reaper? 
  * what happens when you need to trace out CUDA? 
    * make sure the library you are using is "fluentd" compatible...



