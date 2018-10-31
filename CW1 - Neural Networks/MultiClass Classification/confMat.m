function[output, accuracy] = confMat( labels, y)
    
    % check if inputs are the same size
    if sum(size(labels) == size(y)) < 2         % if each dimension is true, then [1 1]. sum = 2
        fprintf("Size of labels and y are not the same!\n");
        return; 
    end
    
    % variables
    sampleNum = size( labels, 2);
    classNum = size( labels, 1);
	output = zeros( classNum, classNum);
    accuracy = 0;
    
    % round
    %y = roundClass(y);  
    
    % match
	for i = 1: sampleNum
		for j = 1: classNum
            for k = 1: classNum
                if labels( j, i) && y( k, i)
                    output( k, j) = output( k, j) + 1;
                end
			end
		end
    end
    
    % accuracy
    for i = 1: classNum
        accuracy = accuracy + output(i,i);
    end
    accuracy = accuracy/sampleNum;
    
    % display
    fprintf("\t\t\tTargets\n");
    fprintf("\t\t\t");
    for i = 1: classNum
        fprintf("%s\t", 64+i);         % class labels, e.g. A  B  C ..
    end
    fprintf("\nOutputs\t");
    for i = 1: classNum
        fprintf("%s\t", 64+i);         % class labels, e.g. A  B  C ..
        for j = 1: classNum
            fprintf("%d\t", output( i, j));
        end
        fprintf("\n\t\t");
    end
    fprintf("Accuracy: %f\n", accuracy);
    
end