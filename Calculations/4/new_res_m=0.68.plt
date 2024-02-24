a0=4.06505
a1=-0.0393136
a2=0.190978
a3=0.0306125
a4=0.00924069
a5=-0.00204632
b0=4.06504
b1=-0.0390116
b2=2.78242
b3=0.0134995
b4=0.13613
b5=0.0278193
f(x)=(a0+a1*x+a2*x*x+a3*x*x*x+a4*x*x*x*x+a5*x*x*x*x*x)/(b0+b1*x+b2*x*x+b3*x*x*x+b4*x*x*x*x+b5*x*x*x*x*x)
#fit f(x) "new_res_m=0.68.txt" u 1:2 via a0,a1,a2,a3,a4,a5,b0,b1,b2,b3,b4,b5
p\
"new_res_m=0.68.txt" w l,\
"y_res.txt" w l