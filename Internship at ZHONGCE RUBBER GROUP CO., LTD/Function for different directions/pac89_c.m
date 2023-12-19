function result=pac89_c(P,Fz,alpha,gamma,kappa,mode)
if mode=='fy'
    result=fy(P,Fz,alpha,gamma);
%     plot(alpha,result)
else if mode=='mz'
        result=mz(P,Fz,alpha,gamma);
%         plot(alpha,result)
    else if mode=='fx'
            result=fx(P,Fz,kappa);
%             plot(kappa,result)
        end
    end
end

    function fy=fy(P,Fz,alpha,gamma)
        x=alpha;
        C=P(18);
        D=(P(1).*Fz+P(2)).*(1-P(15).*gamma.*gamma).*Fz;
        K=P(3).*sin(atan(Fz./P(4)).*2).*(1-P(5).*abs(gamma));
        B=K./C./D;
        Sh=P(8).*Fz+P(9)+P(10).*gamma;
        Sv=P(11).*Fz+P(12)+(P(13).*Fz.*Fz+P(14).*Fz).*gamma;
        x1=x+Sh;
        E=(P(6).*Fz+P(7)).*(1-(((P(16).*gamma)+P(17)).*sign(x1)));
        fy=D.*sin(C.*atan(B.*x1-E.*(B.*x1-atan(B.*x1))))+Sv;
    end

    function mz=mz(P,Fz,alpha,gamma)
        x=alpha;
        C=P(21);
        D=(P(1).*Fz.*Fz+P(2).*Fz).*(1-P(18).*gamma.*gamma);
        K=(P(3).*Fz.*Fz+P(4).*Fz).*(1-(P(6).*abs(gamma))).*exp(-P(5).*Fz);
        B=K./C./D;
        Sh=P(11).*Fz+P(12)+P(13).*gamma;
        Sv=P(14).*Fz+P(15)+(P(16).*Fz.*Fz+P(17).*Fz).*gamma;
        x1=x+Sh;
        E=(P(7).*Fz.*Fz+P(8).*Fz+P(9)).*(1-(P(19).*gamma+P(20)).*sign(x1))./(1-(P(10).*abs(gamma)));
        mz=D.*sin(C.*atan(B.*x1-E.*(B.*x1-atan(B.*x1))))+Sv;
    end

    function fx=fx(P,Fz,kappa)
        x=kappa;
        C=P(14);
        D=P(1).*Fz.*Fz+P(2).*Fz;
        K=(P(3).*Fz.*Fz+P(4).*Fz).*exp(-P(5).*Fz);
        B=K./C./D;
        Sh=P(9).*Fz+P(10);
        Sv=P(11).*Fz+P(12);
        x1=x+Sh;
        E=((P(6).*Fz+P(7)).*Fz+P(8)).*(1-P(13).*sign(x1));
        fx=D.*sin(C.*atan(B.*x1-E.*(B.*x1-atan(B.*x1))))+Sv;        
    end
    
    

end