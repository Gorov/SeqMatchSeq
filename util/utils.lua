--[[
Copyright 2015 Singapore Management University (SMU). All Rights Reserved.

Permission to use, copy, modify and distribute this software and its documentation for purposes of research, teaching and general academic pursuits, without fee and without a signed licensing agreement, is hereby granted, provided that the above copyright statement, this paragraph and the following paragraph on disclaimer appear in all copies, modifications, and distributions.  Contact Singapore Management University, Intellectual Property Management Office at iie@smu.edu.sg, for commercial licensing opportunities.

This software is provided by the copyright holder and creator “as is” and any express or implied warranties, including, but not Limited to, the implied warranties of merchantability and fitness for a particular purpose are disclaimed.  In no event shall SMU or the creator be liable for any direct, indirect, incidental, special, exemplary or consequential damages, however caused arising in any way out of the use of this software.
]]
function share_params(cell, src)
    if torch.type(cell) == 'nn.gModule' then
        for i = 1, #cell.forwardnodes do
            local node = cell.forwardnodes[i]
            if node.data.module then
                node.data.module:share(src.forwardnodes[i].data.module,
                                    'weight', 'bias', 'gradWeight', 'gradBias')
            end
        end
    elseif torch.isTypeOf(cell, 'nn.Module') then
        cell:share(src, 'weight', 'bias', 'gradWeight', 'gradBias')
    else
        error('parameters cannot be shared for this input')
    end
end


function MAP(ground_label, predict_label)
    local map = 0
    local map_idx = 0
    local extracted = {}

    for i = 1, ground_label:size(1) do
        if ground_label[i] ~= 0 then extracted[i] = 1 end
    end

    local val, key = torch.sort(predict_label, 1,true)
    for i = 1, key:size(1) do
        if extracted[key[i]] ~= nil then
            map_idx = map_idx + 1
            map = map + map_idx / i
        end
    end
    assert(map_idx ~= 0)
    map = map / map_idx
    return map
end

function MRR(ground_label, predict_label)
    local mrr = 0
    local map_idx = 0
    local extracted = {}

    for i = 1, ground_label:size(1) do
        if ground_label[i] ~= 0 then extracted[i] = 1 end
    end

    local val, key = torch.sort(predict_label, 1,true)
    for i = 1, key:size(1) do
        if extracted[key[i]] ~= nil then
            mrr = 1.0 / i
            break
        end
    end
    assert(mrr ~= 0)
    return mrr

end

function longSentCut(sent, labels, len)
    local half_len = len / 2
    if sent:size(1) > len then
        if type(labels) == 'table' then
            local cut_start, cut_end
            local span_len = labels[2] - labels[1]
            if labels[1]-half_len > 1 then
                cut_start = labels[1]-half_len
            else
                cut_start = 1
            end
            if labels[2]+half_len > sent:size(1) then
                cut_end = sent:size(1)
            else
                cut_end = labels[2]+half_len
            end
            if cut_start ~= 1 then
                labels[1] = half_len + 1
                labels[2] = half_len + 1 + span_len
            end

            sent = sent:sub(cut_start, cut_end)
        else
            local cut_start, cut_end
            local span_len = labels[-1] - labels[1] + 1
            local add_len = len - span_len
            if add_len <= 0 then
                cut_start = labels[1]
                cut_end = labels[-1]
            else
                add_len = add_len > 100 and add_len or 100
                add_len_pre = add_len - 50

                if labels[1]-add_len_pre > 1 then
                    cut_start = labels[1]-add_len_pre
                else
                    cut_start = 1
                end

                add_len_pos = add_len - (labels[1]-cut_start+1)

                if labels[-1]+add_len_pos > sent:size(1) then
                    cut_end = sent:size(1)
                else
                    cut_end = labels[-1]+add_len_pos
                end
            end

            if cut_start ~= 1 then
                labels:add(1-cut_start)
            end

            assert(labels:min()>0)
            sent = sent:sub(cut_start, cut_end)

        end

    end


    return {sent, labels}
end

function padSent(sents, padVal)
    assert(type(sents) == 'table')
    padVal = padVal or 1
    local sizes = torch.LongTensor(#sents)
    for i, sent in pairs(sents) do
        sizes[i] = sent:size(1)
    end
    local max_size = sizes:max()
    local sents_pad = torch.LongTensor(#sents, max_size):fill(padVal)
    for i, sent in pairs(sents) do
        sents_pad[i]:sub(1, sizes[i]):copy(sents[i])
    end
    return {sents_pad, sizes}

end

function padBSent(sents, sizes_batch, padVal)
    assert(type(sents) == 'table')
    padVal = padVal or 1
    local sent_sizes = torch.LongTensor(#sents)
    local sents_num = torch.LongTensor(#sents)
    for i, sent in pairs(sents) do
        sent_sizes[i] = sent:size(2)
        sents_num[i] = sent:size(1)
    end
    local max_sent_size = sent_sizes:max()
    local max_sents_num = sents_num:max()
    local sents_pad = torch.LongTensor(#sents, max_sents_num, max_sent_size):fill(padVal)
    local sizes = torch.LongTensor(#sents*max_sents_num):fill(0)
    for i, sent in pairs(sents) do
        sents_pad[{i, {1, sents_num[i]}, {1, sent_sizes[i]}}]:copy(sent)
        sizes:sub((i-1)*max_sents_num+1, (i-1)*max_sents_num+sizes_batch[i]:size(1)):copy(sizes_batch[i])
    end

    return {sents_pad, sizes}

end

function samplePara(p_sents_batch, p_sizes_batch, para_idx_batch, labels, labels_b, sample_num)
    local p_sents_batch_new = {}
    local p_sizes_batch_new = {}

    local labels_new = labels.new(labels:size(1)):copy(labels)
    local labels_b_new = labels_b.new(labels_b:size(1)):copy(labels_b)

    local batch_size = #p_sents_batch
    for i, p_sents in pairs(p_sents_batch) do
        p_sizes = p_sizes_batch[i]

        local para_idx = para_idx_batch[i]
        para_idx = 1
        local sizes_sum = 0
        for j = 1, p_sizes:size(1) do
            sizes_sum = sizes_sum + p_sizes[j]
            if sizes_sum > labels[i] then
                para_idx = j~=1 and j - 1 or 1
                break
            end
        end

        local sample_num_new = p_sents:size(1) <= sample_num and p_sents:size(1)-1 or sample_num

        local indices = torch.randperm(p_sents:size(1)):sub(1, sample_num_new+1)

        if indices:eq(para_idx):sum() ~= 1 then indices[sample_num_new+1] = para_idx end
        indices = indices:sort()

        local p_sizes_new = torch.LongTensor(sample_num_new+1)
        for j = 1, sample_num_new+1 do
            p_sizes_new[j] = p_sizes[indices[j]]
        end
        local p_sents_new = p_sents.new(sample_num_new+1, p_sizes_new:max()):fill(1)
        if para_idx ~= 1 then
            labels_new[i] = labels[i] - p_sizes:sub(1,para_idx-1):sum()
            labels_new[batch_size+i] = labels[batch_size+i] - p_sizes:sub(1,para_idx-1):sum()
        end
        for j = 1, sample_num_new+1 do
            p_sents_new[j]:sub(1, p_sizes[indices[j]]):copy(p_sents[indices[j]]:sub(1, p_sizes[indices[j]]))
            if indices[j] < para_idx then
                labels_new[i] = labels_new[i] + p_sizes_new[j]
                labels_new[batch_size+i] = labels_new[batch_size+i] + p_sizes_new[j]
            end
        end
        labels_b_new[i] = labels_new[batch_size+i]
        labels_b_new[batch_size+i] = labels_new[i]
        p_sents_batch_new[i] = p_sents_new
        p_sizes_batch_new[i] = p_sizes_new
        assert(labels_new[i] > 0 and labels_new[batch_size+i] > 0)
    end
    return {p_sents_batch_new, p_sizes_batch_new, labels_new, labels_b_new}

end



function tensorReverse(tensor)
    local output = tensor.new(tensor:size())
    local num = tensor:size(1)
    for i = 1, num do
        output[i] = tensor[num - i + 1]
    end
    return output
end


