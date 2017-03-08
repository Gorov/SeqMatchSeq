--[[
Copyright 2015 Singapore Management University (SMU). All Rights Reserved.

Permission to use, copy, modify and distribute this software and its documentation for purposes of research, teaching and general academic pursuits, without fee and without a signed licensing agreement, is hereby granted, provided that the above copyright statement, this paragraph and the following paragraph on disclaimer appear in all copies, modifications, and distributions.  Contact Singapore Management University, Intellectual Property Management Office at iie@smu.edu.sg, for commercial licensing opportunities.

This software is provided by the copyright holder and creator “as is” and any express or implied warranties, including, but not Limited to, the implied warranties of merchantability and fitness for a particular purpose are disclaimed.  In no event shall SMU or the creator be liable for any direct, indirect, incidental, special, exemplary or consequential damages, however caused arising in any way out of the use of this software.
]]

local compAggWebqspQsPos = torch.class('seqmatchseq.compAggWebqspQsPos')

function compAggWebqspQsPos:__init(config)
    self.mem_dim       = config.mem_dim       or 100
    self.att_dim       = config.att_dim       or self.mem_dim
    self.cov_dim       = config.cov_dim       or self.mem_dim
    self.learning_rate = config.learning_rate or 0.001
    self.batch_size    = config.batch_size    or 25
    self.emb_dim       = config.wvecDim       or 300
    self.task          = config.task          or 'wikiqa'
    self.numWords      = config.numWords
    self.dropoutP      = config.dropoutP      or 0
    self.grad          = config.grad          or 'adamax'
    self.visualize     = false
    self.emb_lr        = config.emb_lr        or 0.001
    self.emb_partial   = config.emb_partial   or true
    self.comp_type      = config.comp_type      or 'concate'
    self.window_sizes  = {1,2,3}
    self.window_large  = self.window_sizes[#self.window_sizes]

    self.best_score    = 0

    self.pos_emb_vecs = nn.LookupTable(200, self.mem_dim):cuda()

    self.as_emb_vecs = nn.LookupTable(self.numWords, self.emb_dim):cuda()
    self.q_emb_vecs = nn.LookupTable(self.numWords, self.emb_dim):cuda()


    self.answer_pool = torch.load('../data/webqsp/answer_pool.t7')

    self.proj_module_master = self:new_proj_module()

    if self.comp_type == 'concate' then
        self.att_module_master = self:new_sim_con_module()
    elseif self.comp_type == 'sub' then
        self.att_module_master = self:new_sim_sub_module()
    elseif self.comp_type == 'mul' then
        self.att_module_master = self:new_sim_mul_module()
    elseif self.comp_type == 'weightsub' then
        self.att_module_master = self:new_sim_weightsub_module()
    elseif self.comp_type == 'weightmul' then
        self.att_module_master = self:new_sim_weightmul_module()
    elseif self.comp_type == 'bilinear' then
        self.att_module_master = self:new_sim_bilinear_module()
    elseif self.comp_type == 'submul' then
        self.att_module_master = self:new_sim_submul_module()
    elseif self.comp_type == 'cos' then
        self.cov_dim = 2
        self.att_module_master = self:new_sim_cos_module()
    else
        error("The word matching method is not provided!!")
    end

    self.conv_module = nn.Sequential()
                       :add(cudnn.LSTM(2*self.mem_dim, self.mem_dim, 1, true))
                       :add(nn.Max(2))
                       --:add(nn.Select(2,-1))

    self.conv_module = self:new_conv_module()


    self.soft_module = nn.Sequential()
                       :add(nn.Linear(self.mem_dim, 1))
                       :add(nn.View(-1))
                       :add(nn.LogSoftMax())

    self.dropout_modules = {}
    self.proj_modules = {}
    self.att_modules = {}

    self.join_module = nn.JoinTable(1)

    self.optim_state = { learningRate = self.learning_rate }
    self.criterion = nn.DistKLDivCriterion():cuda()


    local modules = nn.Parallel()
        :add(self.proj_module_master)
        :add(self.att_module_master)
        :add(self.conv_module)
        :add(self.soft_module)
        :add(self.pos_emb_vecs)
        :add(self.as_emb_vecs)
        :cuda()
    self.params, self.grad_params = modules:getParameters()
    self.best_params = self.params.new(self.params:size())
    print(self.params:size())
    for i = 1, 2 do
        self.proj_modules[i] = self:new_proj_module()
        self.dropout_modules[i] = nn.Dropout(self.dropoutP):cuda()
    end
    self.dropout_modules[3] = nn.Dropout(self.dropoutP):cuda()
    share_params(self.q_emb_vecs, self.as_emb_vecs)
    self.as_emb_vecs.weight:copy( tr:loadVacab2Emb(self.task):float() )
end

function compAggWebqspQsPos:new_proj_module()
    local input = nn.Identity()()
    local i = nn.Sigmoid()(nn.BLinear(self.emb_dim, self.mem_dim)(input))
    local u = nn.Tanh()(nn.BLinear(self.emb_dim, self.mem_dim)(input))
    local output = nn.CMulTable(){i, u}
    local module = nn.gModule({input}, {output}):cuda()
    if self.proj_module_master then
        share_params(module, self.proj_module_master)
    end
    return module
end

function compAggWebqspQsPos:new_att_module()
    local linput, rinput = nn.Identity()(), nn.Identity()()
    --padding
    local lPad = nn.Padding(1,1)(linput)
    --local M_l = nn.Linear(self.mem_dim, self.mem_dim)(lPad)

    local M_r = nn.MM(false, true){lPad, rinput}

    local alpha = nn.SoftMax()( nn.Transpose({1,2})(M_r) )

    local Yl =  nn.MM(){alpha, lPad}

    local att_module = nn.gModule({linput, rinput}, {Yl})

    if self.att_module_master then
        share_params(att_module, self.att_module_master)
    end

    return att_module
end
function compAggWebqspQsPos:new_conv_module()
    local conv_module = nn.DepthConcat(2)

    for i, window_size in pairs(self.window_sizes) do

        local pad_conv = nn.Sequential()
        if window_size ~= 1 then
            pad_conv:add( nn.Padding(2, (window_size-1)/2) )
                    :add( nn.Padding(2, -(window_size-1)/2) )
        end
        pad_conv:add( cudnn.TemporalConvolution(self.mem_dim*3, self.mem_dim, window_size) )
                --:add( nn.View(-1, self.conv_dim) )
                --:add( cudnn.BatchNormalization(self.conv_dim) )
                --:add( nn.View(self.batch_size, -1, self.conv_dim) )
                :add( cudnn.ReLU())
                :add( nn.Max(2) )
        conv_module:add( pad_conv )
    end
    local module = nn.Sequential()
                   :add(nn.JoinTable(3))
                   --:add(nn.BLinear(self.mem_dim*3, self.mem_dim*2)):add(nn.ReLU())
                   :add(conv_module)
                   :add(nn.Dropout(self.dropoutP))
                   :add(nn.Linear(#self.window_sizes * self.mem_dim, self.mem_dim))
                   :add(nn.Tanh())

    return module
end


function compAggWebqspQsPos:new_sim_con_module()
    local pinput, qinput, qsizes = nn.Identity()(), nn.Identity()(), nn.Identity()(), nn.Identity()()

    local qinput_pad = nn.Padding(2,1)(qinput)

    local M_q = nn.BLinear(self.mem_dim, self.mem_dim)(qinput_pad)

    local M_pq = nn.MM(false, true){pinput, M_q}

    local alpha = nn.MaskedSoftMax(){M_pq, qsizes}

    local q_wsum =  nn.MM(){alpha, qinput_pad}

    --local match = nn.Dropout(self.dropoutP)(nn.ReLU()(nn.BLinear(2*self.mem_dim, 2*self.mem_dim)(nn.JoinTable(3){pinput, q_wsum})))
    local match = nn.JoinTable(3){pinput, q_wsum}

    local match_module = nn.gModule({pinput, qinput, qsizes}, {match})

    return match_module
end


function compAggWebqspQsPos:new_sim_submul_module()
    local pinput, qinput, qsizes = nn.Identity()(), nn.Identity()(), nn.Identity()(), nn.Identity()()

    local qinput_pad = nn.Padding(2,1)(qinput)

    local M_q = nn.BLinear(self.mem_dim, self.mem_dim)(qinput_pad)

    local M_pq = nn.MM(false, true){pinput, M_q}

    local alpha = nn.MaskedSoftMax(){M_pq, qsizes}

    local q_wsum =  nn.MM(){alpha, qinput_pad}

    --local match = nn.Dropout(self.dropoutP)(nn.ReLU()(nn.BLinear(2*self.mem_dim, 2*self.mem_dim)(nn.JoinTable(3){pinput, q_wsum})))
    local match = nn.JoinTable(3){nn.CSubTable(){pinput, q_wsum}, nn.CMulTable(){pinput, q_wsum}}

    local match_module = nn.gModule({pinput, qinput, qsizes}, {match})

    return match_module
end

function compAggWebqspQsPos:new_sim_bilinear_module()
    local inputq, inputa = nn.Identity()(), nn.Identity()()
    local output = nn.ReLU()(nn.Bilinear(self.mem_dim, self.mem_dim, self.mem_dim)({inputq, inputa}))
    local module = nn.gModule({inputq, inputa}, {output})
    return module
end

function compAggWebqspQsPos:new_sim_sub_module()
    local inputq, inputa = nn.Identity()(), nn.Identity()()
    local output = nn.Power(2)(nn.CSubTable(){inputq, inputa})
    local module = nn.gModule({inputq, inputa}, {output})
    return module
end

function compAggWebqspQsPos:new_sim_mul_module()
    local inputq, inputa = nn.Identity()(), nn.Identity()()

    local output = nn.CMulTable(){inputq, inputa}

    local module = nn.gModule({inputq, inputa}, {output})
    return module
end
function compAggWebqspQsPos:new_sim_weightsub_module()
    local inputq, inputa = nn.Identity()(), nn.Identity()()
    local output = nn.Power(2)(nn.CSubTable(){nn.Add(self.mem_dim)(nn.CMul(self.mem_dim)(inputq)), nn.Add(self.mem_dim)(nn.CMul(self.mem_dim)(inputa))})
    local module = nn.gModule({inputq, inputa}, {output})
    return module
end

function compAggWebqspQsPos:new_sim_weightmul_module()
    local inputq, inputa = nn.Identity()(), nn.Identity()()

    local output = nn.CMulTable(){nn.Add(self.mem_dim)(nn.CMul(self.mem_dim)(inputq)), nn.Add(self.mem_dim)(nn.CMul(self.mem_dim)(inputa))}

    local module = nn.gModule({inputq, inputa}, {output})
    return module
end

function compAggWebqspQsPos:new_sim_cos_module()
    local inputq, inputa = nn.Identity()(), nn.Identity()()
    local cos = nn.View(-1,1)(nn.CosineDistance(){inputq, inputa})
    local dis = nn.View(-1,1)(nn.PairwiseDistance(2){inputq, inputa})
    local output = nn.JoinTable(2){cos, dis}

    local module = nn.gModule({inputq, inputa}, {output})
    return module
end

function compAggWebqspQsPos:new_match_module()
    local pinput, qinput, qsizes = nn.Identity()(), nn.Identity()(), nn.Identity()(), nn.Identity()()

    local qinput_pad = nn.Padding(2,1)(qinput)

    local M_q = nn.BLinear(self.mem_dim, self.mem_dim)(qinput_pad)

    local M_pq = nn.MM(false, true){pinput, M_q}

    local alpha = nn.MaskedSoftMax(){M_pq, qsizes}

    local q_wsum =  nn.MM(){alpha, qinput_pad}

    --local match = nn.Dropout(self.dropoutP)(nn.ReLU()(nn.BLinear(2*self.mem_dim, 2*self.mem_dim)(nn.JoinTable(3){pinput, q_wsum})))
    local match = nn.JoinTable(3){pinput, q_wsum}

    local match_module = nn.gModule({pinput, qinput, qsizes}, {match})

    return match_module
end

function compAggWebqspQsPos:train(dataset)
    for i = 1, 2 do
        self.proj_modules[i]:training()
        self.dropout_modules[i]:training()
    end
    self.conv_module:training()
    dataset.size = #dataset
    local indices = torch.randperm(dataset.size)

    local zeros = torch.zeros(self.mem_dim)
    for i = 1, dataset.size, self.batch_size do
        xlua.progress(i, dataset.size)
        local batch_size = math.min(i + self.batch_size - 1, dataset.size) - i + 1

        local feval = function(x)
            self.grad_params:zero()
            self.as_emb_vecs.weight[1]:zero()
            local loss = 0
            for j = 1, batch_size do
                local idx = indices[i + j - 1]
                local data_raw = dataset[idx]
                local data_q = data_raw[1]:view(1,-1)
                local data_as = {}
                local label = data_raw[3]:cuda()
                local pos_q = data_raw[4]:cuda()

                local data_as_len = torch.IntTensor( data_raw[2]:size(1))
                --local indices_batch = torch.randperm( data_raw[2]:size(1) )
                for k = 1, data_raw[2]:size(1) do
                    local idx_batch = k --indices_batch[k]
                    data_as[k] = self.answer_pool[data_raw[2][idx_batch]]
                    data_as_len[k] = data_as[k]:size(1)
                    if data_as_len[k] < self.window_large then
                        local as_tmp = torch.Tensor(self.window_large):fill(1)
                        as_tmp:sub(1,data_as_len[k]):copy(data_as[k])
                        data_as[k] = as_tmp
                        data_as_len[k] = self.window_large
                    end
                end

                local as_sents, as_sizes = unpack(padSent(data_as, 1))
                as_sizes = as_sizes:cuda():add(1)

                local a_emb = self.as_emb_vecs:forward(as_sents)
                local q_emb = self.q_emb_vecs:forward(data_q)
                local pos_q_emb = self.pos_emb_vecs:forward(pos_q)

                local inputs_a_emb = self.dropout_modules[1]:forward(a_emb)
                local inputs_q_emb = self.dropout_modules[2]:forward(q_emb)
                local inputs_pos_q_emb = self.dropout_modules[3]:forward(pos_q_emb)

                local inputs_pos_q_emb_rep = torch.repeatTensor(inputs_pos_q_emb, #data_as, 1, 1)

                local projs_a_emb = self.proj_modules[1]:forward(inputs_a_emb)
                local projs_q_emb = self.proj_modules[2]:forward(inputs_q_emb)
                --if data_q:size(1) == 1 then projs_q_emb:resize(1, self.mem_dim) end

                local projs_q_emb_rep = torch.repeatTensor(projs_q_emb, #data_as, 1, 1)

                local att_output = self.att_module_master:forward({projs_q_emb_rep, projs_a_emb, as_sizes})

                local conv_output = self.conv_module:forward({att_output, inputs_pos_q_emb_rep})
                local soft_output = self.soft_module:forward(conv_output)
                local example_loss = self.criterion:forward(soft_output, label)

                loss = loss + example_loss

                local crit_grad = self.criterion:backward(soft_output, label)
                local soft_grad = self.soft_module:backward(conv_output, crit_grad)


                local conv_grad = self.conv_module:backward({att_output, inputs_pos_q_emb_rep}, soft_grad)

                local att_grad = self.att_module_master:backward({projs_q_emb_rep, projs_a_emb, as_sizes}, conv_grad[1])

                local projs_a_emb_grad = self.proj_modules[1]:backward(inputs_a_emb, att_grad[2])
                local projs_q_emb_grad = self.proj_modules[2]:backward(inputs_q_emb, att_grad[1]:sum(1))

                local inputs_pos_q_emb_grad = self.dropout_modules[3]:backward(pos_q_emb, conv_grad[2]:sum(1)[1])
                self.pos_emb_vecs:backward(pos_q, inputs_pos_q_emb_grad)

                local inputs_a_emb_grad = self.dropout_modules[1]:backward(a_emb, projs_a_emb_grad)
                local inputs_q_emb_grad = self.dropout_modules[2]:backward(q_emb, projs_q_emb_grad)

                self.as_emb_vecs:backward(as_sents, inputs_a_emb_grad)
                self.q_emb_vecs:backward(data_q, inputs_q_emb_grad)
            end
            loss = loss / batch_size
            self.grad_params:div(batch_size)
            --print(loss)
            return loss, self.grad_params
        end

        optim[self.grad](feval, self.params, self.optim_state)
        collectgarbage()

    end
    xlua.progress(dataset.size, dataset.size)
end


function compAggWebqspQsPos:predict(data_raw)
    local data_q = data_raw[1]:view(1,-1)
    local data_as = {}
    local label = data_raw[3]:cuda()
    local pos_q = data_raw[4]:cuda()

    local data_as_len = torch.IntTensor( data_raw[2]:size(1))
    --local indices_batch = torch.randperm( data_raw[2]:size(1) )
    for k = 1, data_raw[2]:size(1) do
        local idx_batch = k --indices_batch[k]
        data_as[k] = self.answer_pool[data_raw[2][idx_batch]]
        data_as_len[k] = data_as[k]:size(1)
        if data_as_len[k] < self.window_large then
            local as_tmp = torch.Tensor(self.window_large):fill(1)
            as_tmp:sub(1,data_as_len[k]):copy(data_as[k])
            data_as[k] = as_tmp
            data_as_len[k] = self.window_large
        end
    end

    local as_sents, as_sizes = unpack(padSent(data_as, 1))
    as_sizes = as_sizes:cuda():add(1)

    local inputs_a_emb = self.as_emb_vecs:forward(as_sents)
    local inputs_q_emb = self.q_emb_vecs:forward(data_q)
    local pos_q_emb = self.pos_emb_vecs:forward(pos_q)

    inputs_a_emb = self.dropout_modules[1]:forward(inputs_a_emb)
    inputs_q_emb = self.dropout_modules[2]:forward(inputs_q_emb)
    local inputs_pos_q_emb = self.dropout_modules[3]:forward(pos_q_emb)

    local inputs_pos_q_emb_rep = torch.repeatTensor(inputs_pos_q_emb, #data_as, 1, 1)

    local projs_a_emb = self.proj_modules[1]:forward(inputs_a_emb)
    local projs_q_emb = self.proj_modules[2]:forward(inputs_q_emb)
    --if data_q:size(1) == 1 then projs_q_emb:resize(1, self.mem_dim) end

    local projs_q_emb_rep = torch.repeatTensor(projs_q_emb, #data_as, 1, 1)

    local att_output = self.att_module_master:forward({projs_q_emb_rep, projs_a_emb, as_sizes})

    local conv_output = self.conv_module:forward({att_output, inputs_pos_q_emb_rep})
    local soft_output = self.soft_module:forward(conv_output)
    local _, idx = torch.max(soft_output, 1)

    if label[idx[1]] ~= 0 then
    --if idx[1] == label then
        return 1
    else
        return 0
    end
end

function compAggWebqspQsPos:predict_dataset(dataset)
    for i = 1, 2 do
        self.proj_modules[i]:evaluate()
        self.dropout_modules[i]:evaluate()
    end
    self.conv_module:evaluate()
    local res = {0,0}
    local large_num = 0
    local similar_num = 0
    local large_res = {0,0}
    local similar_res = {0,0}
    local correct_num = 0.0
    dataset.size = #dataset
    for i = 1, dataset.size do
        xlua.progress(i, dataset.size)
        local prediction = self:predict(dataset[i])
        correct_num = correct_num + prediction

    end
    local accuracy = correct_num / dataset.size

    return accuracy
end

function compAggWebqspQsPos:save(path, config, result, epoch)
    assert(string.sub(path,-1,-1)=='/')
    local paraPath = path .. config.task .. config.expIdx
    local paraBestPath = path .. config.task .. config.expIdx .. '_best'
    local recPath = path .. config.task .. config.expIdx ..'Record.txt'

    local file = io.open(recPath, 'a')
    if epoch == 1 then
        for name, val in pairs(config) do
            file:write(name .. '\t' .. tostring(val) ..'\n')
        end
    end

    file:write(config.task..': '..epoch..': ')
    for i, vals in pairs(result) do
        file:write(vals .. ', ')
        if i == 1 then
            print("Dev: accuracy:" .. vals)
        elseif i == 2 then
            print("Test: accuracy:" .. vals)
        else
            print("Train: accuracy:" .. vals)
        end
    end
    file:write('\n')

    file:close()

    if result[1] > self.best_score then
        self.best_score  = result[1]
        self.best_params:copy(self.params)
        torch.save(paraBestPath, {params = self.params,config = config})
    end
    torch.save(paraPath, {params = self.params, config = config})
end

function compAggWebqspQsPos:load(path)
    local state = torch.load(path)
    self:__init(state.config)
    self.params:copy(state.params)
end
