%
% Purpose: Simulate branching for array containing randomly distributed
% special cases
%
% Made by:
%       Even Florenes NTNU 2016
%

%% Initalize simulations

% Set number of full simulations
nSimulations = 20;

% Set number of repetions
nRepetions = 100;

% Set length of array
N = 100;

% Set fixed ratio of special cases
rArray = 0:1/N:0.5;
rebranchesavgR =zeros(1,length(rArray));
rebranchesTotal = zeros(nSimulations,length(rArray));
% Initalize results array
rebranchesSimulation = zeros(1,nRepetions);

%% Simulate
for i = 1:nSimulations

    for j = 1:length(rArray)

        r = rArray(j);

        for k = 1:nRepetions

            % Compute number of special cases
            nSpecial = round(r * N);

            % Create temporary array with all array positions
            aIndex = 1:N;

            % Initalize special positions array
            specialPositions = zeros(1,nSpecial);


            for l = 1:nSpecial
                randomInd = round ( 1 + rand * ( N - l + 1 - 1) );
                specialPositions(l) = aIndex( randomInd );
                aIndex( randomInd ) = [];
            end

            % Initalize simulation array
            a = zeros(1,N);
            a(specialPositions) = 1 ;

            % Counter for number of branches
            rebranches = 0;

            % Display to screen simulation status
            %disp(['Simulate case :', num2str(k)]);

            for l = 1:N

                if l == 1
                    if ( a(l) == 0 )
                        rebranches = rebranches + 1;
                    end
                else
                    if ~isequal(a(l-1),a(l))
                        rebranches = rebranches + 1;
                    end
                end % if j

            end % for j

            rebranchesSimulation(k) = rebranches;

        end % for k
        rebranchesavgR(j) = sum(rebranchesSimulation)/(N);
        %disp(['Average number of rebranches: ', num2str(sum(rebranchesSimulation)/N) ]);
        
        
    end % for j
    
    rebranchesTotal(i,:) = rebranchesavgR(:);


end % for i

%%  Display result to screen

plot( N* rArray, rebranchesTotal),xlabel('Number of special elements'),ylabel('Number of branches');
title('Simulated number of branches of array with 100 elements')

