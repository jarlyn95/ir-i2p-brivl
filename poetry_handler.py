import jsonlines


def read_raw_ccpc(path='data/poetry/CCPC/ccpc_test_v1.0.json'):
    with open(path) as f:
        data = [ item for item in jsonlines.Reader(f)]
    return data


def to_lines(data):
    poetry_content = [poetry['content'].split('|') for poetry in data]
    lines = [line for content in poetry_content for line in content]
    return lines


def get_poetry_by_line(line, data):
    for poetry in data:
        if line in poetry['content']:
            return poetry


if __name__ == '__main__':
    data = []
    test_data = read_raw_ccpc('data/poetry/CCPC/ccpc_test_v1.0.json')
    data.extend(test_data)
    train_data = read_raw_ccpc('data/poetry/CCPC/ccpc_train_v1.0.json')
    data.extend(train_data)
    valid_data = read_raw_ccpc('data/poetry/CCPC/ccpc_valid_v1.0.json')
    data.extend(valid_data)

    lines = to_lines(data)
    char_num_per_line = [len(line) for line in lines]
    print(min(char_num_per_line), max(char_num_per_line))
    poetry = get_poetry_by_line(lines[13], data)
    print(lines[13])
    # poetry = get_poetry_by_line('南岳庐山拄杖头', data)
    print(poetry)
