%Round the highest value in y (output) = 1, in order to represent a class.
%and round the lower values in y(output) = 0.
function [yr] = roundClass(y)
    sampleNum = size( y, 2);
    classNum = size( y, 1);
    yr = zeros( classNum, sampleNum);
    temp = 0;
    pos = 0;
    for i=1:sampleNum
        for j = 1: classNum
            if y( j, i) > temp
                pos = j;
                temp = y( j, i);
            end
        end
        yr( pos, i) = 1;
    end
end

