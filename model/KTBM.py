import torch
import torch.nn as nn

class KTBM(nn.Module):
    """
    Extension of Memory-Augmented Neural Network (MANN)
    """

    def __init__(self, config):
        super(KTBM, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        self.cuda = self.is_cuda & config.cuda
        if self.cuda:
            torch.cuda.manual_seed(config.seed)
            self.device = torch.device("cuda")
            # torch.cuda.set_device(config.gpu_device)
        else:
            torch.cuda.manual_seed(config.seed)
            self.device = torch.device("cpu")
        self.metric = config.metric

        # initialize the dim size hyper parameters
        self.num_questions = config.num_items
        self.num_nongradable_items = config.num_nongradable_items
        self.embedding_size_q = config.embedding_size_q
        self.embedding_size_a = config.embedding_size_a
        self.embedding_size_l = config.embedding_size_l
        self.embedding_size_d = config.embedding_size_d
        self.embedding_size_q_behavior = config.embedding_size_q_behavior
        self.embedding_size_l_behavior = config.embedding_size_l_behavior

        self.num_concepts = config.num_concepts
        self.key_dim = config.key_dim
        self.value_dim = config.value_dim
        self.summary_dim = config.summary_dim
        self.init_std = config.init_std

        self.behavior_summary_fc = config.behavior_summary_fc
        self.behavior_map_size = config.behavior_map_size
        self.behavior_hidden_size = config.behavior_hidden_size


        # initialize the embedding layers
        self.q_embed_matrix = nn.Embedding(num_embeddings=self.num_questions + 1,
                                           embedding_dim=self.embedding_size_q,
                                           padding_idx=0)

        self.l_embed_matrix = nn.Embedding(num_embeddings=self.num_nongradable_items + 1,
                                           embedding_dim=self.embedding_size_l,
                                           padding_idx=0)

        if self.metric == "rmse":
            self.a_embed_matrix = nn.Linear(1, self.embedding_size_a)
        else:
            self.a_embed_matrix = nn.Embedding(3, self.embedding_size_a, padding_idx=2)


        self.d_embed_matrix = nn.Embedding(3, self.embedding_size_d, padding_idx=2)
        self.q_behavior_embed_matrix = nn.Embedding(num_embeddings=self.num_questions + 1,
                                           embedding_dim=self.embedding_size_q_behavior,
                                           padding_idx=0)

        self.l_behavior_embed_matrix = nn.Embedding(num_embeddings=self.num_nongradable_items + 1,
                                           embedding_dim=self.embedding_size_l_behavior,
                                           padding_idx=0)


        # initialize the knowledge module layers
        self.init_knowledge_module()

        # initialize the behavior module layers
        self.init_behavior_module()


        # initialize the activiate functions
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()

    def init_knowledge_module(self):
        # initialize the parameters
        self.key_matrix = torch.Tensor(self.num_concepts, self.key_dim).to(self.device)
        nn.init.normal_(self.key_matrix, mean=0, std=self.init_std)

        self.value_matrix_init = torch.Tensor(self.num_concepts, self.value_dim).to(self.device)
        nn.init.normal_(self.value_matrix_init, mean=0., std=self.init_std)

        # initialize the MANN layers
        self.mapQ_value = nn.Linear(self.embedding_size_q + self.embedding_size_a, self.value_dim)
        self.mapL_value = nn.Linear(self.embedding_size_l, self.value_dim)

        self.mapQ_key = nn.Linear(self.embedding_size_q, self.key_dim)
        self.mapL_key = nn.Linear(self.embedding_size_l, self.key_dim)

        self.mapQ_key_type = nn.Linear(self.embedding_size_q, self.key_dim)
        self.mapL_key_type = nn.Linear(self.embedding_size_l, self.key_dim)

        self.erase_E_Q = nn.Linear(self.embedding_size_q + self.embedding_size_a, self.value_dim, bias=True)
        self.erase_E_L = nn.Linear(self.embedding_size_l, self.value_dim, bias=True)
        self.erase_E_bh = nn.Linear(self.behavior_hidden_size, self.value_dim, bias=False)


        self.add_D_Q = nn.Linear(self.embedding_size_q + self.embedding_size_a, self.value_dim, bias=True)
        self.add_D_L = nn.Linear(self.embedding_size_l, self.value_dim, bias=True)
        self.add_D_bh = nn.Linear(self.behavior_hidden_size, self.value_dim, bias=False)

        self.T_QQ = nn.Linear(self.value_dim, self.value_dim, bias=False)
        self.T_QL = nn.Linear(self.value_dim, self.value_dim, bias=False)
        self.T_LQ = nn.Linear(self.value_dim, self.value_dim, bias=False)
        self.T_LL = nn.Linear(self.value_dim, self.value_dim, bias=False)

        self.summary_fc = nn.Linear(self.embedding_size_q + self.value_dim, self.summary_dim)
        self.linear_out = nn.Linear(self.summary_dim, 1)
        self.linear_out_type_q = nn.Linear(self.value_dim, 1)
        # self.linear_out_type_l = nn.Linear(self.value_dim, 1)


    def init_behavior_module(self):
        # initialize the LSTM layers
        self.behavior_mapQ = nn.Linear(self.embedding_size_q_behavior + self.embedding_size_d, self.behavior_map_size, bias=True)
        self.behavior_mapL = nn.Linear(self.embedding_size_l_behavior + self.embedding_size_d, self.behavior_map_size, bias=True)
        self.behavior_mapKnowledge = nn.Linear(self.value_dim, self.num_concepts, bias=True)

        self.sum_knowledge2behavior = nn.Linear(self.num_concepts, 1, bias=True)

        self.W_i = nn.Linear(self.behavior_map_size, self.behavior_hidden_size, bias=True)
        self.W_i_knowledge = nn.Linear(self.value_dim, self.behavior_hidden_size, bias=True)
        self.W_ih = nn.Linear(self.behavior_hidden_size, self.behavior_hidden_size, bias=False)


        self.W_g = nn.Linear(self.behavior_map_size, self.behavior_hidden_size, bias=True)
        self.W_g_knowledge = nn.Linear(self.value_dim, self.behavior_hidden_size, bias=True)
        self.W_gh = nn.Linear(self.behavior_hidden_size, self.behavior_hidden_size, bias=False)


        self.W_f = nn.Linear(self.behavior_map_size, self.behavior_hidden_size, bias=True)
        self.W_f_knowledge = nn.Linear(self.value_dim, self.behavior_hidden_size, bias=True)
        self.W_fh = nn.Linear(self.behavior_hidden_size, self.behavior_hidden_size, bias=False)


        self.W_o = nn.Linear(self.behavior_map_size, self.behavior_hidden_size, bias=True)
        self.W_o_knowledge = nn.Linear(self.value_dim, self.behavior_hidden_size, bias=True)
        self.W_oh = nn.Linear(self.behavior_hidden_size, self.behavior_hidden_size, bias=False)


        self.behavior_prefrence_Q = nn.Linear(self.behavior_hidden_size + self.embedding_size_d + self.embedding_size_q_behavior, self.behavior_summary_fc, bias=True)
        self.behavior_prefrence_L = nn.Linear(self.behavior_hidden_size + self.embedding_size_d + self.embedding_size_l_behavior, self.behavior_summary_fc, bias=True)
        self.behavior_out_type = nn.Linear(self.behavior_summary_fc, 1, bias=True)


    def forward(self, q_data, a_data, l_data, d_data):
        '''
           get output of the model
           :param q_data: (batch_size, seq_len) question indexes/ids of each learning interaction, 0 represent paddings
           :param a_data: (batch_size, seq_len) student performance of each learning interaction, 2 represent paddings
           :param l_data: (batch_size, seq_len) non-assessed material indexes/ids of each learning interaction, 0 represent paddings
           :param d_data: (batch_size, seq_len) material type of each learning interaction, 0: question 1:non-assessed material
           :return:
       '''

        # inintialize M^v
        batch_size, seq_len = q_data.size(0), q_data.size(1)
        # inintialize h0 and m0 and value matrix
        self.h = torch.zeros(batch_size, self.behavior_hidden_size).to(self.device)
        self.m = torch.zeros(batch_size, self.behavior_hidden_size).to(self.device)
        self.value_matrix = self.value_matrix_init.clone().repeat(batch_size, 1, 1)

        # get embeddings of learning material and response
        q_embed_data = self.q_embed_matrix(q_data.long())
        if self.metric == 'rmse':
            a_data = torch.unsqueeze(a_data, dim=2)
            a_embed_data = self.a_embed_matrix(a_data)
        else:
            a_embed_data = self.a_embed_matrix(a_data)
        l_embed_data = self.l_embed_matrix(l_data)
        d_embed_data = self.d_embed_matrix(d_data)
        q_behavior_embed_data = self.q_behavior_embed_matrix(q_data)
        l_behavior_embed_data = self.l_behavior_embed_matrix(l_data)

        # split the data seq into chunk and process each question sequentially, and get embeddings of each learning
        # materia
        sliced_q_embed_data = torch.chunk(q_embed_data, seq_len, dim=1)
        sliced_a_embed_data = torch.chunk(a_embed_data, seq_len, dim=1)
        sliced_l_embed_data = torch.chunk(l_embed_data, seq_len, dim=1)
        sliced_d_embed_data = torch.chunk(d_embed_data, seq_len, dim=1)
        sliced_q_behavior_embed_data = torch.chunk(q_behavior_embed_data, seq_len, dim=1)
        sliced_l_behavior_embed_data = torch.chunk(l_behavior_embed_data, seq_len, dim=1)

        sliced_d_data = torch.chunk(d_data, seq_len, dim=1)

        batch_pred, batch_pred_type = [], []

        for i in range(1, seq_len - 1):
            # embedding layer, get material embeddings and neighbors embeddings for each time step t
            q = sliced_q_embed_data[i].squeeze(1)  # (batch_size, emebeding_size_q)
            a = sliced_a_embed_data[i].squeeze(1)
            l = sliced_l_embed_data[i].squeeze(1)
            d = sliced_d_embed_data[i].squeeze(1)
            q_b = sliced_q_behavior_embed_data[i].squeeze(1)
            l_b = sliced_l_behavior_embed_data[i].squeeze(1)
            d_t = sliced_d_data[i]
            d_t_1 = sliced_d_data[i - 1]

            #updata knowledge state
            self.knowledge_MANN(q, a, l, d_t, d_t_1)

            #update behavior prefrence
            self.behavior_LSTM(q_b, d, l_b, d_t)

            #predict type
            prefrence_type = (1 - d_t)*self.behavior_prefrence_Q(torch.cat([q_b, d, self.h], dim = 1)) + d_t*self.behavior_prefrence_L(torch.cat([l_b, d, self.h], dim=1))
            batch_sliced_pred_type= self.sigmoid(self.behavior_out_type(prefrence_type))
            batch_pred_type.append(batch_sliced_pred_type)

            #predict response
            q_next = sliced_q_embed_data[i + 1].squeeze(1)  # (batch_size, key_dim)
            q_read_key_next = self.mapQ_key(q_next)
            correlation_weight_next = self.compute_correlation_weight(q_read_key_next)
            read_content_next = self.read(correlation_weight_next, d_t)

            mastery_level = torch.cat([read_content_next, q_next], dim=1)
            summary_output = self.tanh(self.summary_fc(mastery_level))
            batch_sliced_pred = self.sigmoid(self.linear_out(summary_output))
            batch_pred.append(batch_sliced_pred)

        batch_pred = torch.cat(batch_pred, dim=-1)
        batch_pred_type = torch.cat(batch_pred_type, dim=-1)
        return batch_pred, batch_pred_type


    def knowledge_MANN(self, q, a, l, d_t, d_t_1):
        qa = torch.cat([q, a], dim=1)
        q_read_key = self.mapQ_key(q)
        l_read_key = self.mapL_key(l)

        lnmt_embedded = (
                                1 - d_t) * q_read_key + d_t * l_read_key  # learning material embedding mapped to concept for getting knowledge

        correlation_weight = self.compute_correlation_weight(lnmt_embedded)

        self.value_matrix = self.write(correlation_weight, qa, l, d_t, d_t_1)


    def behavior_LSTM(self, q, d, l, d_t):
        qd = torch.cat([q, d], dim = 1)
        ld = torch.cat([l, d], dim = 1)
        maped_embedded = (1 - d_t)*self.behavior_mapQ(qd) + d_t*self.behavior_mapL(ld)
        sumed_knowledge = self.sum_knowledge2behavior(torch.transpose(self.value_matrix, 1,2)).squeeze(2)

        i = self.sigmoid(self.W_i(maped_embedded) + self.W_ih(self.h) + self.W_i_knowledge(sumed_knowledge))
        g = self.tanh(self.W_g(maped_embedded) + self.W_gh(self.h) + self.W_g_knowledge(sumed_knowledge))
        f = self.sigmoid(self.W_f(maped_embedded) + self.W_fh(self.h) + self.W_f_knowledge(sumed_knowledge))
        o = self.sigmoid(self.W_o(maped_embedded) + self.W_oh(self.h) + self.W_o_knowledge(sumed_knowledge))

        self.m = f * self.m + i * g
        self.h = o * self.tanh(self.m)



    def compute_correlation_weight(self, query_embedded):
        """
        use dot product to find the similarity between question embedding and each concept
        embedding stored as key_matrix
        where key-matrix could be understood as all concept embedding covered by the course.
        query_embeded : (batch_size, concept_embedding_dim)
        key_matrix : (num_concepts, concept_embedding_dim)
        output: is the correlation distribution between question and all concepts
        """

        similarity = query_embedded @ self.key_matrix.t()
        correlation_weight = torch.softmax(similarity, dim=1)
        return correlation_weight

    def read(self, correlation_weight, d_t):
        """
        read function is to read a student's knowledge level on part of concepts covered by a
        target question.
        we could view value_matrix as the latent representation of a student's knowledge
        in terms of all possible concepts.

        value_matrix: (batch_size, num_concepts, concept_embedding_dim)
        correlation_weight: (batch_size, num_concepts)
        """
        batch_size = self.value_matrix.size(0)
        value_matrix_reshaped = torch.transpose(
            d_t * torch.transpose(self.T_QQ(self.value_matrix), 0, 1) + (1 - d_t) * torch.transpose(self.T_LQ(
                self.value_matrix), 0, 1), 0, 1)
        value_matrix_reshaped = value_matrix_reshaped.reshape(
            batch_size * self.num_concepts, self.value_dim
        )
        correlation_weight_reshaped = correlation_weight.reshape(batch_size * self.num_concepts, 1)
        # a (10,3) * b (10,1) = c (10, 3)is every row vector of a multiplies the row scalar of b
        # the multiplication below is to scale the memory embedding by the correlation weight
        rc = value_matrix_reshaped * correlation_weight_reshaped
        read_content = rc.reshape(batch_size, self.num_concepts, self.value_dim)
        read_content = torch.sum(read_content, dim=1)  # sum over all concepts

        return read_content

    def write(self, correlation_weight, qa_embed, l_embed, d_t, d_t_1):
        """
                write function is to update memory based on the interaction
                value_matrix: (batch_size, memory_size, memory_state_dim)
                correlation_weight: (batch_size, memory_size)
                qa_embedded: (batch_size, memory_state_dim)
                """
        batch_size = self.value_matrix.size(0)
        erase_vector = self.sigmoid(
            (1 - d_t) * self.erase_E_Q(qa_embed) + d_t * self.erase_E_L(l_embed) + self.erase_E_bh(self.h))  # (batch_size, value_dim)

        add_vector = self.tanh(
            (1 - d_t) * self.add_D_Q(qa_embed) + d_t * self.add_D_L(l_embed) + self.add_D_bh(self.h))  # (batch_size, value_dim)

        erase_reshaped = erase_vector.reshape(batch_size, 1, self.value_dim)
        cw_reshaped = correlation_weight.reshape(batch_size, self.num_concepts,
                                                 1)  # the multiplication is to generate weighted erase vector for each memory cell, therefore, the size is (batch_size, num_concepts, value_dim)
        erase_mul = erase_reshaped * cw_reshaped
        # memory_after_erase = self.value_matrix * (1 - erase_mul)

        memory_after_erase = torch.transpose(
            (((1 - d_t) * (1 - d_t_1)) * torch.transpose(self.T_QQ(self.value_matrix), 0, 1)) + (
                        d_t * d_t_1) * torch.transpose(self.T_LL(
                self.value_matrix), 0, 1) + ((1 - d_t_1) * d_t) * torch.transpose(self.T_QL(self.value_matrix), 0,
                                                                                  1) + (
                        d_t_1 * (1 - d_t)) * torch.transpose(self.T_LQ(
                self.value_matrix), 0, 1), 0, 1) * (1 - erase_mul)

        add_reshaped = add_vector.reshape(batch_size, 1,
                                          self.value_dim)  # the multiplication is to generate weighted add vector for each memory cell therefore, the size is (batch_size, num_concepts, value_dim)
        add_memory = add_reshaped * cw_reshaped
        updated_memory = memory_after_erase + add_memory

        return updated_memory

    def read_type(self, correlation_weight):
        """
        read function is to read a student's knowledge level on part of concepts covered by a
        target question.
        we could view value_matrix as the latent representation of a student's knowledge
        in terms of all possible concepts.
        value_matrix: (batch_size, num_concepts, concept_embedding_dim)
        correlation_weight: (batch_size, num_concepts)
        """
        batch_size = self.value_matrix.size(0)
        value_matrix_reshaped = self.value_matrix.reshape(
            batch_size * self.num_concepts, self.value_dim
        )
        correlation_weight_reshaped = correlation_weight.reshape(batch_size * self.num_concepts, 1)
        # a (10,3) * b (10,1) = c (10, 3)is every row vector of a multiplies the row scalar of b
        # the multiplication below is to scale the memory embedding by the correlation weight
        rc = value_matrix_reshaped * correlation_weight_reshaped
        read_content = rc.reshape(batch_size, self.num_concepts, self.value_dim)
        read_content = torch.sum(read_content, dim=1)  # sum over all concepts

        return read_content
