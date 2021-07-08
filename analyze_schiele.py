with open('schiele.txt') as f:
    names = set()
    for line in f:
        if ':' in line:
            name = line.split(':')[0]
            if not name in names:
                names.add(name)
    
    with open('schiele_names.txt', 'w') as names_f:
        for name in names:
            names_f.write(name + '\n')