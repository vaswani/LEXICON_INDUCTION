function list = split(s,d)
    list = {};
    while (length(s) > 0)
        [t,s] = strtok(s,d);
        list = {list{:}, t};
    end
end