a0=198.791
a1=100.669
a2=52.4641
a3=15.9206
a4=-5.48707
b0=198.795
b1=100.517
b2=180.69
b3=75.1294
b4=32.9556
f(x)=(a0+a1*x+a2*x*x+a3*x*x*x+a4*x*x*x*x)/(b0+b1*x+b2*x*x+b3*x*x*x+b4*x*x*x*x)
#fit f(x) "new_res_m=0.68.txt" u 1:2 via a0,a1,a2,a3,a4,b0,b1,b2,b3,b4

p\
"new_res_m=0.68.txt" w l,\
f(x) w l