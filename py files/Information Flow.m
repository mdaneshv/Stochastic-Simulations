N = 50;    % number of cells
M = 500;    % number of time slices
m = 1;    % number of time-steps between slices
T = M*m;    % total number of time-steps = T = m * M
znew = zeros(1,N);     % lattice configuration intermediate to make one time-step
z_aver = zeros(M+1,N);   % averaged lattice configuration initialized to 0
randval = zeros(1,N+1);   % array of random values initilazed to 0

s = randi(5,1,N);    % number of servers in each queue(line) (max=5)
max_rate = 10;    % maximum rate of transfer of one customer between queues

delta_t = 1/50;
MC=5000;    % the number of simulations (trajectories)

for j=1:MC

    %z = randi(some_num,1,N);    % random initial configuration
    z = zeros(1,N);    % zero initial configuration
    
    % creating an evolution matrix
    for num_slice = 1:M
        for i = 1:m
            
            % BEGIN one time-step
            znew = z;
            randval = rand(1,N+1);
           
            % periodic arrival rate
            if randval(1) <= (6+5*sin(0.1*((num_slice-1)*m+i))) * delta_t
                znew(1)=z(1)+1;
            end
            
            % transition k -> k+1
            for k=1:N-1
                
                if randval(k+1) <= max_rate * v(z(k),s(k)) * delta_t
                    znew(k)   = znew(k) - 1;
                    znew(k+1) = znew(k+1) + 1;
                end
            end
            
            % boundary condition
            % exiting
            if randval(N+1) <= max_rate * v(z(N),s(N)) * delta_t
                znew(N) = znew(N) - 1;
            end
            
            z = znew;
            
        end
        
        z_aver(num_slice+1,:) = z_aver(num_slice+1,:) + z;
        
    end
    
end

z_aver = z_aver / MC;

t = linspace(0, T*delta_t,M+1);
x = linspace(0,N,N);
zmax = max(max(z_aver(:,1)),max(z_aver(:,2)));

figure;
for time=0:m:T
    t = linspace(0, time*delta_t, time/m);
    clf;
    tmpstr = 'Time = sub1';
    plot(t, z_aver(1:time, 1),'b');
    hold on
    plot(t, z_aver(1:time, 2),'m');
    xlim([0 T*delta_t])
    ylim([0 zmax])
    legend(['first queue: num of servers=',sprintf('%d',s(1))],...
        ['second queue: num of servers=',sprintf('%d',s(2))],'Location','SouthEast');
    
    title('time evolution for the first and second queue until stationary');
    xlabel('time');
    ylabel('average of customers');
    ax = gca;
    set(ax,'FontSize',12)
    pause(0.01)
end


% throttling function
function throttling = v(x,s)

if x < s
    throttling = x/s;
else
    throttling = 1;
end
end
