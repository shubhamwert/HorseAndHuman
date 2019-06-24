function K = selectK(S)
    K=1;
    total=0
    SX=sum(sum(S));
    fract=total/SX;
    fprintf('Calculating K ..');

    while fract<0.99
        fprintf('.');
        total=total+S(K,K);
        fract=total/SX;
        K++;
    end
    



end