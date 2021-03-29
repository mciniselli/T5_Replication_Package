one_char = ["!", "#", "%", "&", "\\", "'", "(", ")", "*", "+", "-", ".", "/",
            ":", ";", "<", "=", ">", "?", "[", "]", "^", "{", "|", "}", "~", ","]
# they must be ordered by length decreasing
multi_char = ["...", "::", ">>=", "<<=", ">>>", "<<<", "===", "!=", "==", "===", "<=", ">=", "*=", "&&", "-=", "+=",
              "||", "--", "++", "|=", "&=", "%=", "/=", "^=", "::"]


# it counts the number of slash before a specific position (used to check if " or ' are escaped or not)
def count_slash(ss, position):
    count = 0;
    pos = position - 1
    try:
        while ss[pos] == "\\":
            count += 1
            pos -= 1
    except:
        pass
    return count


# given a line of code, it retrieves the start end the end position for each string
def get_strings(code):
    start = list()
    end = list()

    start_string = False

    for i, c in enumerate(code):
        if c == "\"":
            num_slash = count_slash(code, i)
            if num_slash % 2 == 0 and start_string:
                end.append(i)
                start_string = False
            elif num_slash % 2 == 0:
                start.append(i)
                start_string = True

    return start, end


# given a line of code, it retrieves the start end the end position for each char (e.g. 'c')
def get_chars(code):
    start = list()
    end = list()

    start_string = False

    for i, c in enumerate(code):
        if c == "'":
            num_slash = count_slash(code, i)
            if num_slash % 2 == 0 and start_string:
                end.append(i)
                start_string = False
            elif num_slash % 2 == 0:
                start.append(i)
                start_string = True

    return start, end


# tokenizer (given a line of code, it returns the list of tokens)
def tokenize(code):
    mask = code.replace("<z>", "")

    s, e = get_strings(mask)

    dict_string = dict()
    dict_chars = dict()

    delay = 0

    # replace each string with a placeholder
    for i, (a, b) in enumerate(zip(s, e)):
        dict_string[i] = mask[a:b + 1]

        mask = mask[:a + delay] + " ___STRING___" + str(i) + "__ " + mask[b + 1 + delay:]

        delay = len(" ___STRING___" + str(i) + "__ ") - (b + 1 - a)

    s, e = get_chars(mask)

    # replace each char with a placeholder
    for i, (a, b) in enumerate(zip(s, e)):
        dict_chars[i] = mask[a:b + 1]

        mask = mask[:a + delay] + " ___CHARS___" + str(i) + "__ " + mask[b + 1 + delay:]

        delay = len(" ___CHARS___" + str(i) + "__ ") - (b + 1 - a)

    # replace each char with multiple chars with a placeholder
    for i, c in enumerate(multi_char):
        mask = mask.replace(c, " " + "__ID__" + str(i) + "__ ")

    # add a space before and after each char
    for c in one_char:

        if c == ".":  # it should be a number (0.09)
            index_ = [i for i, ltr in enumerate(mask) if ltr == c]
            # print(index_)
            for i in index_:
                try:
                    if mask[i + 1].isnumeric():
                        continue
                    else:
                        mask = mask.replace(c, " " + c + " ")
                except:
                    pass
        else:
            mask = mask.replace(c, " " + c + " ")

    # remove all multichars' placeholders
    for i, c in enumerate(multi_char):
        mask = mask.replace("__ID__" + str(i) + "__", c)

    # removing double spaces
    while "  " in mask:
        mask = mask.replace("  ", " ")

    mask = mask.strip()
    # retrieving the list of tokens
    tokens = mask.split(" ")

    for t in tokens:
        try:
            if "___STRING___" in t:
                curr = t.replace("___STRING___", "").replace("__", "")
                t = dict_string[int(curr)]
            if "___CHARS___" in t:
                curr = t.replace("___CHARS___", "").replace("__", "")
                t = dict_chars[int(curr)]
        except:
            pass

    # if len(mask.split(" ")) != int(real_length[key]):
    # print(len(mask.split(" ")), real_length[temp[0]])
    # print(code, mask, mask.split(" "))

    return tokens
