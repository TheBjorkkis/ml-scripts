function solve = sudoku(sudo)

    % Define all of the spots which were preset in sudoku.
    predef = zeros(size(sudo,1), size(sudo,2));
    predef(sudo ~= 0) = 1;
    
    maxLength = size(sudo,2);
    n = sqrt(maxLength);    % The size of block's side.
    
    stillSolving = true;
    X = 1;
    Y = 1;
    
    % Repeat the process before the end is reached or the sudoku isn't
    % solvable
    tic;
    while stillSolving

        % Unsolvable.
        if Y <= 0
            stillSolving = false;
      
        % End of the line.
        elseif X > maxLength && Y < maxLength
            X = 1;
            Y = Y + 1; 
            
        % If the sudoku is solved.
        elseif X > maxLength && Y == maxLength
           stillSolving = false; 
     
        % If the value is predefined.
        elseif predef( X, Y ) == 1
            X = X + 1;      
        
        % Reverse to the previous index and reset current index.
        elseif sudo( X, Y ) > maxLength
            sudo( X, Y ) = 0;
            
            if X > 1
                X = X - 1;
                
            else
                Y = Y - 1;
                X = maxLength;
                
            end
            
            % If the sudoku isn't solvable.
            if Y == 0
                break;
                
            end
            
            % If the current index is predefined:
            % skip all of the predefined values.
            while predef( X, Y ) ~= 0
                
                if X > 1
                    X = X - 1;
                    
                else
                    Y = Y - 1;
                    X = maxLength;
                end
            
            end
            

        % Insert number.
        else
            
            % Search for the block where index is located.
            beginX = 1;
            endX = n;
            beginY = 1;
            endY = n;
            unsolveX = true;
            unsolveY = true;
            
            % Range of block in X.
            while unsolveX
                if X >= beginX && X <= endX
                    unsolveX = false;
                else
                    beginX = beginX + n;
                    endX = endX + n;
                end
                
            end
            
            % Range of block in Y.
            while unsolveY
                if Y >= beginY && Y <= endY
                    unsolveY = false;
                else
                    beginY = beginY + n;
                    endY = endY + n;
                end
                
            end
            
            % Define the block where the index is located.
            block = sudo( beginX : endX, beginY : endY );
        
            sudo( X, Y ) = sudo( X, Y ) + 1;
            
            % Raise number until there aren't duplicates 
            % on same row/column/block.
            while numel( find( sudo( X, : ) == sudo( X, Y ) ) ) > 1 || ...
                  numel( find( sudo( :, Y ) == sudo( X, Y ) ) ) > 1 || ...
                  numel( find( block == sudo( X ,Y ) ) ) > 1
                  
                sudo( X, Y ) = sudo( X, Y ) + 1;
                block = sudo( beginX : endX, beginY : endY );
            end
            
            %If number exceeds allowed range.
            if sudo( X, Y ) <= maxLength
                X = X + 1;
            end
        end
        
    end   
    
    % If the sudoku wasn't solvable.
    if numel( find( sudo == 0 ) ) > 0
        disp( "Unsolvable" );
        
    else
        disp( "Solvable" );
        
    end
    
    % Solve the final solution.
    solve = uint8(sudo)
    toc;
    

end
