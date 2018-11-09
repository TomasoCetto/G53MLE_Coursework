function label = majority_value(targets)
    n=0;
    p=0;
    for i:length(targets):
        if targets(i) == 1:
            p++;
        else
            n++;
        end
    end

    if n<p:
        label=1;
    else:
        label=0;