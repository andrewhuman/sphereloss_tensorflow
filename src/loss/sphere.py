import tensorflow as tf

def sphereloss(inputs,label,classes,batch_size,fraction = 1, scope='Logits',reuse=None,m =4,eplion = 1e-8):
    """
    inputs tensor shape=[batch,features_num]
    labels tensor shape=[batch] each unit belong num_outputs

    """
    inputs_shape = inputs.get_shape().as_list()
    with tf.variable_scope(name_or_scope=scope):
        weight = tf.Variable(initial_value=tf.random_normal((classes,inputs_shape[1])) * tf.sqrt(2 / inputs_shape[1]),dtype=tf.float32,name='weights') # shaep =classes, features,
        print("weight shape = ",weight.get_shape().as_list())

    weight_unit = tf.nn.l2_normalize(weight,dim=1)
    print("weight_unit shape = ",weight_unit.get_shape().as_list())

    inputs_mo = tf.sqrt(tf.reduce_sum(tf.square(inputs),axis=1)+eplion) #shape=[batch
    print("inputs_mo shape = ",inputs_mo.get_shape().as_list())

    inputs_unit = tf.nn.l2_normalize(inputs,dim=1)  #shape = [batch,features_num]
    print("inputs_unit shape = ",inputs_unit.get_shape().as_list())

    logits = tf.matmul(inputs,tf.transpose(weight_unit)) #shape = [batch,classes] x * w_unit
    print("logits shape = ",logits.get_shape().as_list())

    weight_unit_batch = tf.gather(weight_unit,label) # shaep =batch,features_num,
    print("weight_unit_batch shape = ",weight_unit_batch.get_shape().as_list())

    logits_inputs = tf.reduce_sum(tf.multiply(inputs,weight_unit_batch),axis=1) # shaep =batch,

    print("logits_inputs shape = ",logits_inputs.get_shape().as_list())

    cos_theta = tf.reduce_sum(tf.multiply(inputs_unit,weight_unit_batch),axis=1) # shaep =batch,
    print("cos_theta shape = ",cos_theta.get_shape().as_list())

    cos_theta_square = tf.square(cos_theta)
    cos_theta_biq = tf.pow(cos_theta,4)
    sign0 = tf.sign(cos_theta)
    sign2 = tf.sign(2 * cos_theta_square-1)
    sign3 = tf.multiply(sign2,sign0)
    sign4 = 2 * sign0 +sign3 -3
    cos_far_theta = sign3 * (8 * cos_theta_biq - 8 * cos_theta_square + 1) + sign4
    print("cos_far_theta  = ",cos_far_theta.get_shape().as_list())

    logit_ii = tf.multiply(cos_far_theta,inputs_mo)#shape = batch
    print("logit_ii shape = ",logit_ii.get_shape().as_list())

    index_range = tf.range(start=0,limit= tf.shape(inputs,out_type=tf.int64)[0],delta=1,dtype=tf.int64)
    index_labels = tf.stack([index_range, label], axis = 1)
    index_logits =  tf.scatter_nd(index_labels,tf.subtract(logit_ii,logits_inputs), tf.shape(logits,out_type=tf.int64))
    print("index_logits shape = ",logit_ii.get_shape().as_list())

    logits_final  = tf.add(logits,index_logits)
    logits_final = fraction * logits_final + (1 - fraction) * logits


    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label,logits=logits_final))

    return logits_final,loss


def soft_loss(inputs,label,classes,scope='Logits'):
    """
    inputs tensor shape=[batch,features_num]
    labels tensor shape=[batch] each unit belong num_outputs

    """
    inputs_shape = inputs.get_shape().as_list()
    with tf.variable_scope(name_or_scope=scope):
        weight = tf.Variable(initial_value=tf.random_normal((classes,inputs_shape[1])) * tf.sqrt(2 / inputs_shape[1]),
                             dtype=tf.float32,name='weights') # shaep =classes, features,
        bias = tf.Variable(initial_value=tf.zeros(classes),dtype=tf.float32,name='bias')
        print("weight shape = ",weight.get_shape().as_list())
        print("bias shape = ", bias.get_shape().as_list())

    weight = tf.Print(weight, [tf.shape(weight)], message='logits weights shape = ',summarize=4, first_n=1)
    logits = tf.nn.bias_add(tf.matmul(inputs,tf.transpose(weight)),bias,name='logits')
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label,logits=logits,
                                                                         name='cross_entropy_per_example'),
                          name='cross_entropy')

    return  logits,loss

def soft_loss_nobias(inputs,label,classes,scope='Logits'):
    """
    inputs tensor shape=[batch,features_num]
    labels tensor shape=[batch] each unit belong num_outputs

    """
    inputs_shape = inputs.get_shape().as_list()
    with tf.variable_scope(name_or_scope=scope):
        weight = tf.Variable(initial_value=tf.random_normal((classes,inputs_shape[1])) * tf.sqrt(2 / inputs_shape[1]),
                             dtype=tf.float32,name='weights') # shaep =classes, features,
        print("weight shape = ",weight.get_shape().as_list())

    weight = tf.Print(weight, [tf.shape(weight)], message='logits weights shape = ',summarize=4, first_n=1)
    logits =tf.matmul(inputs,tf.transpose(weight),name='logits')
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label,logits=logits,
                                                                         name='cross_entropy_per_example'),
                          name='cross_entropy')

    return  logits,loss
