import taichi as ti
ti.init(arch=ti.cuda)

@ti.kernel
def kern():
  for i in range(10):
    print(i)
  
kern()