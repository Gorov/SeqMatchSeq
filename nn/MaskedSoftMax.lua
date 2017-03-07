--[[
Copyright 2015 Singapore Management University (SMU). All Rights Reserved.

Permission to use, copy, modify and distribute this software and its documentation for purposes of research, teaching and general academic pursuits, without fee and without a signed licensing agreement, is hereby granted, provided that the above copyright statement, this paragraph and the following paragraph on disclaimer appear in all copies, modifications, and distributions.  Contact Singapore Management University, Intellectual Property Management Office at iie@smu.edu.sg, for commercial licensing opportunities.

This software is provided by the copyright holder and creator “as is” and any express or implied warranties, including, but not Limited to, the implied warranties of merchantability and fitness for a particular purpose are disclaimed.  In no event shall SMU or the creator be liable for any direct, indirect, incidental, special, exemplary or consequential damages, however caused arising in any way out of the use of this software.
]]
local MaskedSoftMax, parent = torch.class('nn.MaskedSoftMax', 'nn.Module')

function MaskedSoftMax:__init(gpu)
    parent.__init(self)
    self.gradInput = {self.gradInput, self.gradInput.new()}
    self.gpu       = gpu or 1
    self.soft_modules = {}
end

function MaskedSoftMax:updateOutput(input)
    local t_raw = input[1]
    local sizes = input[2]

    assert(t_raw:dim() == 3)
    assert(t_raw:size(1) == sizes:size(1))

    local sizes_max = sizes:max()
    assert(t_raw:size(3) == sizes_max)
    local t = t_raw:size(3) == sizes_max and t_raw or t_raw:view(t_raw:size(1), -1, sizes_max)

    self.output:resizeAs(t):zero()

    for i = 1, t:size(1) do
        local soft_module = self.soft_modules[i]
        if soft_module == nil then
            self.soft_modules[i] = self.gpu == 1 and nn.SoftMax():cuda() or nn.SoftMax()
            soft_module = self.soft_modules[i]
        end
        local output = soft_module:forward(t[{i, {}, {1, sizes[i]}}])
        self.output[{i, {}, {1, sizes[i]}}]:copy( output )
    end
    return self.output
end

function MaskedSoftMax:updateGradInput(input, gradOutput)
    local t_raw = input[1]
    local sizes = input[2]

    local sizes_max = sizes:max()
    local t = t_raw:size(3) == sizes_max and t_raw or t_raw:view(t_raw:size(1), -1, sizes_max)

    self.gradInput[1]:resizeAs(t):zero()
    self.gradInput[2]:resize(sizes:size()):zero()
    for i = 1, t:size(1) do
        local soft_module = self.soft_modules[i]
        local grad = soft_module:backward(t[{i, {}, {1, sizes[i]}}], gradOutput[{i, {}, {1, sizes[i]}}])

        self.gradInput[1][{i, {}, {1, sizes[i]}}]:copy(grad)
    end
    return self.gradInput
end

function MaskedSoftMax:clearState()
    self.gradInput[1]:set()
    self.gradInput[2]:set()
    self.output:set()
    return self
end
