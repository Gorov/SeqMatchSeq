local stringx = require 'pl.stringx'
function buildDict()

    local vocab = {}
    local ivocab = {}
    vocab['<ZEROPADDING>'] = 1
    ivocab[1] = '<ZEROPADDING>'

    local line_num = 0
    for line in io.lines('sequence/relations.words.with_name.txt') do
        if line_num ~= 0 then
            local divs = stringx.split(line, '\t')
            local words = stringx.split(divs[2], ' ')
            for i = 1, #words do
                if words[i]:sub(1, 1) ~= '#' then
                    if vocab[words[i]] == nil then
                        vocab[words[i]] = #ivocab + 1
                        ivocab[#ivocab + 1] = words[i]
                    end
                end
            end
        end
        line_num = line_num + 1
    end


    local filenames = {'sequence/WebQSP.RE.dev.with_boundary.withpool.dlnlp.txt',
                       'sequence/WebQSP.RE.test.with_boundary.withpool.dlnlp.txt',
                       'sequence/WebQSP.RE.train.with_boundary.withpool.dlnlp.txt'}

    for _, filename in pairs(filenames) do
        local line_num = 0
        for line in io.lines(filename) do
            if line_num ~= 0 then
                local divs = stringx.split(line, '\t')
                local words = stringx.split(divs[3], ' ')
                for i = 2, #words-1 do
                    if vocab[words[i]] == nil then
                        vocab[words[i]] = #ivocab + 1
                        ivocab[#ivocab + 1] = words[i]
                    end
                end
            end
            line_num = line_num + 1
        end
    end
    print(#ivocab)
    torch.save('vocab.t7', vocab)
    torch.save('ivocab.t7', ivocab)
end
function buildVacab2Emb()

    local vocab = torch.load("vocab.t7")
    local ivocab = torch.load("ivocab.t7")
    local emb = torch.randn(#ivocab, 300) * 0.01
    local file = io.open("../../data/glove/glove.840B.300d.txt", 'r')

    local count = 0
    local embRec = {}
    local iembRec = {}
    while true do
        local line = file:read()

        if line == nil then break end
        vals = stringx.split(line, ' ')
        if vocab[vals[1]] ~= nil then
            for i = 2, #vals do
                emb[vocab[vals[1]]][i-1] = tonumber(vals[i])
            end
            embRec[vocab[vals[1]]] = #iembRec + 1
            iembRec[#iembRec + 1] = vocab[vals[1]]
            count = count + 1
            if count == #ivocab then
                break
            end
        end
    end
    print("Number of words not appear in glove: "..(#ivocab-count) )
    --self:initialUNK(embRec, emb, opt)
    torch.save("./initEmb.t7", emb)
    torch.save("./unUpdateVocab.t7", embRec)
    torch.save("./unUpdateiVocab.t7", iembRec)

end
function empty (table)
    for _, _ in pairs(table) do
        return false
    end
    return true
end
function buildData()
    local vocab = torch.load("vocab.t7")
    local ivocab = torch.load("ivocab.t7")

    local data_train = {}
    local answer_pool = {}
    local line_num = 0
    for line in io.lines('sequence/relations.words.with_name.txt') do
        if line_num ~= 0 then
            local divs = stringx.split(line, '\t')
            local answer_idx = tonumber(divs[1])
            local words = stringx.split(divs[2], ' ')
            local answer = {}
            for i = 1, #words do
                if words[i]:sub(1, 1) ~= '#' then
                    answer[#answer+1] = vocab[words[i]]
                end
            end
            if #answer == 0 then answer[1] = 1 print (line_num) print(divs[2]) end

            answer_pool[answer_idx] = torch.LongTensor(answer)

        end
        line_num = line_num + 1
    end

    torch.save('answer_pool.t7', answer_pool)


    local filenames = {train = 'sequence/WebQSP.RE.dev.with_boundary.withpool.dlnlp.txt',
                       dev  = 'sequence/WebQSP.RE.test.with_boundary.withpool.dlnlp.txt'}


    for f, filename in pairs(filenames) do
        local data = {}
        local line_num = 0
        for line in io.lines(filename) do
            local question = {}
            local candidate = {}
            if line_num ~= 0 then
                local divs = stringx.split(line, '\t')
                local nums = stringx.split(divs[1], ' ')
                local answer_idx = {}
                for i = 1, #nums do answer_idx[tonumber(nums[i])] = i end

                local cand_idx = stringx.split(divs[2], ' ')
                local label = torch.FloatTensor(#cand_idx):zero()

                for i = 1, #cand_idx do
                    candidate[#candidate + 1] = tonumber(cand_idx[i])
                    if answer_idx[candidate[#candidate]] ~= nil then
                        label[i] = 1.0 / #nums
                    end
                end

                local words = stringx.split(divs[3], ' ')
                local en_pos = 1
                for i = 2, #words-1 do
                    question[#question + 1] = vocab[words[i]]
                    if words[i] == '<e>' then
                        en_pos = i - 1
                    end
                end
                local pos = torch.range(1, #words - 2) - en_pos + 50

                data[#data + 1] = {torch.LongTensor(question), torch.LongTensor(candidate), label, pos}
            end
            line_num = line_num + 1
        end
        --[[
        if f == 'train' then
            local indices = torch.randperm(#data)
            local dev_num_instance = torch.ceil(#data / 5)
            local data_train = {}
            local data_dev = {}
            for i = 1, #data do
                if i <= dev_num_instance then
                    data_dev[#data_dev + 1] = data[indices[i] ]
                else
                    data_train[#data_train + 1] = data[indices[i] ]
                end
            end

            torch.save('sequence/dev.t7', data_dev)
            torch.save('sequence/train.t7', data_train)
        else
            torch.save('sequence/'..f..'.t7', data)
        end
        ]]
        torch.save('sequence/'..f..'.t7', data)
    end

end


--buildDict()
--buildVacab2Emb()
buildData()
