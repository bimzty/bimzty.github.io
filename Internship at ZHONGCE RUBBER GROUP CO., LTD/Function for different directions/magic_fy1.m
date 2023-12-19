function y=magic_fy1(P,x,Fz_nol,Fz0,gamma ) %units 常数 角度 牛 弧度
dfz1=(Fz_nol-Fz0)/Fz0;
shy=(P(12)+P(13)*dfz1)+P(14)*gamma;
x=x+shy;
uy=(P(2)+P(3)*dfz1).*(1-P(4)*gamma.*gamma);
Cy=P(1);
Dy=uy.*Fz_nol;

Ky0=P(9)*Fz0*sin(2*atan(Fz_nol/P(10)/Fz0));
Ky=Ky0.*(1-P(11)*gamma);

svy=Fz_nol.*((P(15)+P(16)*dfz1)+(P(17)+P(18)*dfz1).*gamma);
Ky_gamma_0=P(14)*Ky0+Fz_nol.*(P(17)*P(18)*dfz1);
By=Ky./Cy./Dy;
Ey=(P(5)+P(6)*dfz1).*(1-(P(7)+P(8)*gamma).*sign(x));
L=Ey>1;      %%取出大于1的数的逻辑值              
    N=Ey.*L;     %%取出大于1的数的数
    M=Ey+L;      %%原数组大于1的数加1
    Ey=M-N;      %%原数组大于1的数减去原数（即为1） 
%Fy(i)=Dy*sin(Cy*atan(By*x(i)-Ey*(By*x(i)-atan(By*x(i)))))+svy;
y=Dy.*sin(Cy.*atan(By.*x-Ey.*(By.*x-atan(By.*x))))+svy;
end