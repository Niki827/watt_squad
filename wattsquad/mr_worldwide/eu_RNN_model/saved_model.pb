ш▓.
 т
D
AddV2
x"T
y"T
z"T"
Ttype:
2	АР
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( И
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
>
Maximum
x"T
y"T
z"T"
Ttype:
2	
Ж
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( И
?
Mul
x"T
y"T
z"T"
Ttype:
2	Р

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
│
PartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetypeИ
@
RealDiv
x"T
y"T
z"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype
-
Sqrt
x"T
y"T"
Ttype:

2
┴
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring Ии
Ъ
StatelessWhile

input2T
output2T"
T
list(type)("
condfunc"
bodyfunc" 
output_shapeslist(shape)
 "
parallel_iterationsint

@
StaticRegexFullMatch	
input

output
"
patternstring
ў
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
-
Tanh
x"T
y"T"
Ttype:

2
░
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type/
output_handleКщшelement_dtype"
element_dtypetype"

shape_typetype:
2	
Я
TensorListReserve
element_shape"
shape_type
num_elements(
handleКщшelement_dtype"
element_dtypetype"

shape_typetype:
2	
И
TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsint         
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И"serve*2.10.02unknown8┴А-
В
ConstConst*"
_output_shapes
:	*
dtype0*A
value8B6	"$hЎc@;F@зЧ▓?1f=║ЁoAxДUA°=@гZh7█▓╟7
Д
Const_1Const*"
_output_shapes
:	*
dtype0*A
value8B6	"$eB@лєК@ZШ▓>qУ.;Дж5BЪЦlBП/@Q  ?l  >
Н
Adam/lstm/lstm_cell/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*+
shared_nameAdam/lstm/lstm_cell/bias/v
Ж
.Adam/lstm/lstm_cell/bias/v/Read/ReadVariableOpReadVariableOpAdam/lstm/lstm_cell/bias/v*
_output_shapes	
:А*
dtype0
й
&Adam/lstm/lstm_cell/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@А*7
shared_name(&Adam/lstm/lstm_cell/recurrent_kernel/v
в
:Adam/lstm/lstm_cell/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp&Adam/lstm/lstm_cell/recurrent_kernel/v*
_output_shapes
:	@А*
dtype0
Х
Adam/lstm/lstm_cell/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:		А*-
shared_nameAdam/lstm/lstm_cell/kernel/v
О
0Adam/lstm/lstm_cell/kernel/v/Read/ReadVariableOpReadVariableOpAdam/lstm/lstm_cell/kernel/v*
_output_shapes
:		А*
dtype0
{
Adam/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:э*"
shared_nameAdam/dense/bias/v
t
%Adam/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense/bias/v*
_output_shapes	
:э*
dtype0
Г
Adam/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@э*$
shared_nameAdam/dense/kernel/v
|
'Adam/dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/v*
_output_shapes
:	@э*
dtype0
Н
Adam/lstm/lstm_cell/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*+
shared_nameAdam/lstm/lstm_cell/bias/m
Ж
.Adam/lstm/lstm_cell/bias/m/Read/ReadVariableOpReadVariableOpAdam/lstm/lstm_cell/bias/m*
_output_shapes	
:А*
dtype0
й
&Adam/lstm/lstm_cell/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@А*7
shared_name(&Adam/lstm/lstm_cell/recurrent_kernel/m
в
:Adam/lstm/lstm_cell/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp&Adam/lstm/lstm_cell/recurrent_kernel/m*
_output_shapes
:	@А*
dtype0
Х
Adam/lstm/lstm_cell/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:		А*-
shared_nameAdam/lstm/lstm_cell/kernel/m
О
0Adam/lstm/lstm_cell/kernel/m/Read/ReadVariableOpReadVariableOpAdam/lstm/lstm_cell/kernel/m*
_output_shapes
:		А*
dtype0
{
Adam/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:э*"
shared_nameAdam/dense/bias/m
t
%Adam/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense/bias/m*
_output_shapes	
:э*
dtype0
Г
Adam/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@э*$
shared_nameAdam/dense/kernel/m
|
'Adam/dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/m*
_output_shapes
:	@э*
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	

lstm/lstm_cell/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*$
shared_namelstm/lstm_cell/bias
x
'lstm/lstm_cell/bias/Read/ReadVariableOpReadVariableOplstm/lstm_cell/bias*
_output_shapes	
:А*
dtype0
Ы
lstm/lstm_cell/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@А*0
shared_name!lstm/lstm_cell/recurrent_kernel
Ф
3lstm/lstm_cell/recurrent_kernel/Read/ReadVariableOpReadVariableOplstm/lstm_cell/recurrent_kernel*
_output_shapes
:	@А*
dtype0
З
lstm/lstm_cell/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:		А*&
shared_namelstm/lstm_cell/kernel
А
)lstm/lstm_cell/kernel/Read/ReadVariableOpReadVariableOplstm/lstm_cell/kernel*
_output_shapes
:		А*
dtype0
m

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:э*
shared_name
dense/bias
f
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes	
:э*
dtype0
u
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@э*
shared_namedense/kernel
n
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes
:	@э*
dtype0
b
count_2VarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	count_2
[
count_2/Read/ReadVariableOpReadVariableOpcount_2*
_output_shapes
: *
dtype0	
h
varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*
shared_name
variance
a
variance/Read/ReadVariableOpReadVariableOpvariance*
_output_shapes
:	*
dtype0
`
meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*
shared_namemean
Y
mean/Read/ReadVariableOpReadVariableOpmean*
_output_shapes
:	*
dtype0
▓
#serving_default_normalization_inputPlaceholder*=
_output_shapes+
):'                           *
dtype0*2
shape):'                           
┴
StatefulPartitionedCallStatefulPartitionedCall#serving_default_normalization_inputConstConst_1lstm/lstm_cell/kernellstm/lstm_cell/recurrent_kernellstm/lstm_cell/biasdense/kernel
dense/bias*
Tin

2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         э*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8В *,
f'R%
#__inference_signature_wrapper_22230

NoOpNoOp
ц3
Const_2Const"/device:CPU:0*
_output_shapes
: *
dtype0*Я3
valueХ3BТ3 BЛ3
┴
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*	&call_and_return_all_conditional_losses

_default_save_signature
	optimizer

signatures*
╛
	keras_api

_keep_axis
_reduce_axis
_reduce_axis_mask
_broadcast_shape
mean

adapt_mean
variance
adapt_variance
	count
_adapt_function*
┴
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_random_generator
cell

state_spec*
ж
	variables
 trainable_variables
!regularization_losses
"	keras_api
#__call__
*$&call_and_return_all_conditional_losses

%kernel
&bias*
<
0
1
2
'3
(4
)5
%6
&7*
'
'0
(1
)2
%3
&4*
* 
░
*non_trainable_variables

+layers
,metrics
-layer_regularization_losses
.layer_metrics
	variables
trainable_variables
regularization_losses
__call__

_default_save_signature
*	&call_and_return_all_conditional_losses
&	"call_and_return_conditional_losses*
6
/trace_0
0trace_1
1trace_2
2trace_3* 
6
3trace_0
4trace_1
5trace_2
6trace_3* 
 
7	capture_0
8	capture_1* 
Ю
9iter

:beta_1

;beta_2
	<decay
=learning_rate%mm&mn'mo(mp)mq%vr&vs'vt(vu)vv*

>serving_default* 
* 
* 
* 
* 
* 
RL
VARIABLE_VALUEmean4layer_with_weights-0/mean/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEvariance8layer_with_weights-0/variance/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEcount_25layer_with_weights-0/count/.ATTRIBUTES/VARIABLE_VALUE*

?trace_0* 

'0
(1
)2*

'0
(1
)2*
* 
Я

@states
Anon_trainable_variables

Blayers
Cmetrics
Dlayer_regularization_losses
Elayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
Ftrace_0
Gtrace_1
Htrace_2
Itrace_3* 
6
Jtrace_0
Ktrace_1
Ltrace_2
Mtrace_3* 
* 
у
N	variables
Otrainable_variables
Pregularization_losses
Q	keras_api
R__call__
*S&call_and_return_all_conditional_losses
T_random_generator
U
state_size

'kernel
(recurrent_kernel
)bias*
* 

%0
&1*

%0
&1*
* 
У
Vnon_trainable_variables

Wlayers
Xmetrics
Ylayer_regularization_losses
Zlayer_metrics
	variables
 trainable_variables
!regularization_losses
#__call__
*$&call_and_return_all_conditional_losses
&$"call_and_return_conditional_losses*

[trace_0* 

\trace_0* 
\V
VARIABLE_VALUEdense/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
dense/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUElstm/lstm_cell/kernel&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUElstm/lstm_cell/recurrent_kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUElstm/lstm_cell/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*

0
1
2*

0
1
2*

]0
^1*
* 
* 
 
7	capture_0
8	capture_1* 
 
7	capture_0
8	capture_1* 
 
7	capture_0
8	capture_1* 
 
7	capture_0
8	capture_1* 
 
7	capture_0
8	capture_1* 
 
7	capture_0
8	capture_1* 
 
7	capture_0
8	capture_1* 
 
7	capture_0
8	capture_1* 
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
 
7	capture_0
8	capture_1* 
* 
* 
* 

0*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

'0
(1
)2*

'0
(1
)2*
* 
У
_non_trainable_variables

`layers
ametrics
blayer_regularization_losses
clayer_metrics
N	variables
Otrainable_variables
Pregularization_losses
R__call__
*S&call_and_return_all_conditional_losses
&S"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
8
d	variables
e	keras_api
	ftotal
	gcount*
H
h	variables
i	keras_api
	jtotal
	kcount
l
_fn_kwargs*
* 
* 
* 
* 
* 

f0
g1*

d	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

j0
k1*

h	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
y
VARIABLE_VALUEAdam/dense/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/dense/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUEAdam/lstm/lstm_cell/kernel/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
В|
VARIABLE_VALUE&Adam/lstm/lstm_cell/recurrent_kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUEAdam/lstm/lstm_cell/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/dense/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUEAdam/lstm/lstm_cell/kernel/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
В|
VARIABLE_VALUE&Adam/lstm/lstm_cell/recurrent_kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUEAdam/lstm/lstm_cell/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
╠

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamemean/Read/ReadVariableOpvariance/Read/ReadVariableOpcount_2/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp)lstm/lstm_cell/kernel/Read/ReadVariableOp3lstm/lstm_cell/recurrent_kernel/Read/ReadVariableOp'lstm/lstm_cell/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp'Adam/dense/kernel/m/Read/ReadVariableOp%Adam/dense/bias/m/Read/ReadVariableOp0Adam/lstm/lstm_cell/kernel/m/Read/ReadVariableOp:Adam/lstm/lstm_cell/recurrent_kernel/m/Read/ReadVariableOp.Adam/lstm/lstm_cell/bias/m/Read/ReadVariableOp'Adam/dense/kernel/v/Read/ReadVariableOp%Adam/dense/bias/v/Read/ReadVariableOp0Adam/lstm/lstm_cell/kernel/v/Read/ReadVariableOp:Adam/lstm/lstm_cell/recurrent_kernel/v/Read/ReadVariableOp.Adam/lstm/lstm_cell/bias/v/Read/ReadVariableOpConst_2*(
Tin!
2		*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *'
f"R 
__inference__traced_save_25082
й
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamemeanvariancecount_2dense/kernel
dense/biaslstm/lstm_cell/kernellstm/lstm_cell/recurrent_kernellstm/lstm_cell/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotal_1count_1totalcountAdam/dense/kernel/mAdam/dense/bias/mAdam/lstm/lstm_cell/kernel/m&Adam/lstm/lstm_cell/recurrent_kernel/mAdam/lstm/lstm_cell/bias/mAdam/dense/kernel/vAdam/dense/bias/vAdam/lstm/lstm_cell/kernel/v&Adam/lstm/lstm_cell/recurrent_kernel/vAdam/lstm/lstm_cell/bias/v*'
Tin 
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В **
f%R#
!__inference__traced_restore_25173√А,
╪)
╬
while_body_19885
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_bias_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel#
while_matmul_1_recurrent_kernel
while_biasadd_biasИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    	   ж
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         	*
element_dtype0Т
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*(
_output_shapes
:         АГ
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*(
_output_shapes
:         Аw
	while/addAddV2while/MatMul:product:0while/MatMul_1:product:0*
T0*(
_output_shapes
:         Аp
while/BiasAddBiasAddwhile/add:z:0while_biasadd_bias_0*
T0*(
_output_shapes
:         АW
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :╚
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*`
_output_shapesN
L:         @:         @:         @:         @*
	num_split`
while/SigmoidSigmoidwhile/split:output:0*
T0*'
_output_shapes
:         @b
while/Sigmoid_1Sigmoidwhile/split:output:1*
T0*'
_output_shapes
:         @l
	while/mulMulwhile/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:         @Z

while/TanhTanhwhile/split:output:2*
T0*'
_output_shapes
:         @g
while/mul_1Mulwhile/Sigmoid:y:0while/Tanh:y:0*
T0*'
_output_shapes
:         @f
while/add_1AddV2while/mul:z:0while/mul_1:z:0*
T0*'
_output_shapes
:         @b
while/Sigmoid_2Sigmoidwhile/split:output:3*
T0*'
_output_shapes
:         @W
while/Tanh_1Tanhwhile/add_1:z:0*
T0*'
_output_shapes
:         @k
while/mul_2Mulwhile/Sigmoid_2:y:0while/Tanh_1:y:0*
T0*'
_output_shapes
:         @r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : р
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/mul_2:z:0*
_output_shapes
: *
element_dtype0:щш╥O
while/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :`
while/add_2AddV2while_placeholderwhile/add_2/y:output:0*
T0*
_output_shapes
: O
while/add_3/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_3AddV2while_while_loop_counterwhile/add_3/y:output:0*
T0*
_output_shapes
: L
while/IdentityIdentitywhile/add_3:z:0*
T0*
_output_shapes
: ]
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: N
while/Identity_2Identitywhile/add_2:z:0*
T0*
_output_shapes
: y
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: _
while/Identity_4Identitywhile/mul_2:z:0*
T0*'
_output_shapes
:         @_
while/Identity_5Identitywhile/add_1:z:0*
T0*'
_output_shapes
:         @"*
while_biasadd_biaswhile_biasadd_bias_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O: : : : :         @:         @: : :		А:	@А:А: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         @:-)
'
_output_shapes
:         @:

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:		А:%	!

_output_shapes
:	@А:!


_output_shapes	
:А
ёK
а
&__forward_gpu_lstm_with_fallback_20241

inputs
init_h_0
init_c_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4
strided_slice
cudnnrnn

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
expanddims_1
concat_1
transpose_perm

init_h

init_c
concat_1_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
transpose_7_perm
transpose_8_perm
split_2_split_dim
split_split_dim
split_1_split_dim
concat_axisИc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          B
transpose_0	Transposeinputstranspose/perm:output:0*
T0P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : q

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0*+
_output_shapes
:         @R
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : u
ExpandDims_1
ExpandDimsinit_c_0ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:         @Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :И
splitSplitsplit/split_dim:output:0kernel*
T0*<
_output_shapes*
(:	@:	@:	@:	@*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ц
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*<
_output_shapes*
(:@@:@@:@@:@@*
	num_splitY

zeros_likeConst*
_output_shapes	
:А*
dtype0*
valueBА*    M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : r
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes	
:АS
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Э
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*D
_output_shapes2
0:@:@:@:@:@:@:@:@*
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
         a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes

:@	Y
ReshapeReshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:└a
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes

:@	[
	Reshape_1Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:└a
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

:@	[
	Reshape_2Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:└a
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes

:@	[
	Reshape_3Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:└a
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:@@[
	Reshape_4Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:А a
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0*
_output_shapes

:@@[
	Reshape_5Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:А a
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0*
_output_shapes

:@@[
	Reshape_6Reshapetranspose_7:y:0Const:output:0*
T0*
_output_shapes	
:А a
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0*
_output_shapes

:@@[
	Reshape_7Reshapetranspose_8:y:0Const:output:0*
T0*
_output_shapes	
:А [
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
:@[
	Reshape_9Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
:@\

Reshape_10Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
:@\

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
:@\

Reshape_12Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
:@\

Reshape_13Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
:@\

Reshape_14Reshapesplit_2:output:6Const:output:0*
T0*
_output_shapes
:@\

Reshape_15Reshapesplit_2:output:7Const:output:0*
T0*
_output_shapes
:@O
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Д

concat_1_0ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0╫
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1_0:output:0*
T0*f
_output_shapesT
R:                  @:         @:         @:f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
         _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╛
strided_slice_0StridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_maskp
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:         @*
squeeze_dims
 r
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*'
_output_shapes
:         @*
squeeze_dims
 R
ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B :Е
ExpandDims_2
ExpandDimsstrided_slice_0:output:0ExpandDims_2/dim:output:0*
T0*+
_output_shapes
:         @[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @`
IdentityIdentitystrided_slice_0:output:0*
T0*'
_output_shapes
:         @c

Identity_1IdentityExpandDims_2:output:0*
T0*+
_output_shapes
:         @Z

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:         @\

Identity_3IdentitySqueeze_1:output:0*
T0*'
_output_shapes
:         @I

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
concat_1concat_1_0:output:0"'
concat_1_axisconcat_1/axis:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output_h:0"!

cudnnrnn_0CudnnRNN:output_c:0"

cudnnrnn_1CudnnRNN:output:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"!

expanddimsExpandDims:output:0"%
expanddims_1ExpandDims_1:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
init_cinit_c_0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0")
strided_slicestrided_slice_0:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0"-
transpose_7_permtranspose_7/perm:output:0"-
transpose_8_permtranspose_8/perm:output:0")
transpose_permtranspose/perm:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:                  	:         @:         @:		А:	@А:А*=
api_implements+)lstm_59b087b6-9d64-470e-a94e-2d14718ac25a*
api_preferred_deviceGPU*U
backward_function_name;9__inference___backward_gpu_lstm_with_fallback_20066_20242*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :                  	
 
_user_specified_nameinputs:OK
'
_output_shapes
:         @
 
_user_specified_nameinit_h:OK
'
_output_shapes
:         @
 
_user_specified_nameinit_c:GC

_output_shapes
:		А
 
_user_specified_namekernel:QM

_output_shapes
:	@А
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:А

_user_specified_namebias
Ў:
л
__inference__traced_save_25082
file_prefix#
savev2_mean_read_readvariableop'
#savev2_variance_read_readvariableop&
"savev2_count_2_read_readvariableop	+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop4
0savev2_lstm_lstm_cell_kernel_read_readvariableop>
:savev2_lstm_lstm_cell_recurrent_kernel_read_readvariableop2
.savev2_lstm_lstm_cell_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop2
.savev2_adam_dense_kernel_m_read_readvariableop0
,savev2_adam_dense_bias_m_read_readvariableop;
7savev2_adam_lstm_lstm_cell_kernel_m_read_readvariableopE
Asavev2_adam_lstm_lstm_cell_recurrent_kernel_m_read_readvariableop9
5savev2_adam_lstm_lstm_cell_bias_m_read_readvariableop2
.savev2_adam_dense_kernel_v_read_readvariableop0
,savev2_adam_dense_bias_v_read_readvariableop;
7savev2_adam_lstm_lstm_cell_kernel_v_read_readvariableopE
Asavev2_adam_lstm_lstm_cell_recurrent_kernel_v_read_readvariableop9
5savev2_adam_lstm_lstm_cell_bias_v_read_readvariableop
savev2_const_2

identity_1ИвMergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/partБ
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : У
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ╕
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*с
value╫B╘B4layer_with_weights-0/mean/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-0/variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-0/count/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHе
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*K
valueBB@B B B B B B B B B B B B B B B B B B B B B B B B B B B B Я
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0savev2_mean_read_readvariableop#savev2_variance_read_readvariableop"savev2_count_2_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop0savev2_lstm_lstm_cell_kernel_read_readvariableop:savev2_lstm_lstm_cell_recurrent_kernel_read_readvariableop.savev2_lstm_lstm_cell_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop.savev2_adam_dense_kernel_m_read_readvariableop,savev2_adam_dense_bias_m_read_readvariableop7savev2_adam_lstm_lstm_cell_kernel_m_read_readvariableopAsavev2_adam_lstm_lstm_cell_recurrent_kernel_m_read_readvariableop5savev2_adam_lstm_lstm_cell_bias_m_read_readvariableop.savev2_adam_dense_kernel_v_read_readvariableop,savev2_adam_dense_bias_v_read_readvariableop7savev2_adam_lstm_lstm_cell_kernel_v_read_readvariableopAsavev2_adam_lstm_lstm_cell_recurrent_kernel_v_read_readvariableop5savev2_adam_lstm_lstm_cell_bias_v_read_readvariableopsavev2_const_2"/device:CPU:0*
_output_shapes
 **
dtypes 
2		Р
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:Л
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*╞
_input_shapes┤
▒: :	:	: :	@э:э:		А:	@А:А: : : : : : : : : :	@э:э:		А:	@А:А:	@э:э:		А:	@А:А: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix: 

_output_shapes
:	: 

_output_shapes
:	:

_output_shapes
: :%!

_output_shapes
:	@э:!

_output_shapes	
:э:%!

_output_shapes
:		А:%!

_output_shapes
:	@А:!

_output_shapes	
:А:	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	@э:!

_output_shapes	
:э:%!

_output_shapes
:		А:%!

_output_shapes
:	@А:!

_output_shapes	
:А:%!

_output_shapes
:	@э:!

_output_shapes	
:э:%!

_output_shapes
:		А:%!

_output_shapes
:	@А:!

_output_shapes	
:А:

_output_shapes
: 
╗;
┐
__inference_standard_lstm_24255

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :                  	B
ShapeShapetranspose:y:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ▓
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    	   р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:щ
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         	*
shrink_axis_maske
MatMulMatMulstrided_slice_1:output:0kernel*
T0*(
_output_shapes
:         А_
MatMul_1MatMulinit_hrecurrent_kernel*
T0*(
_output_shapes
:         Аe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         АT
BiasAddBiasAddadd:z:0bias*
T0*(
_output_shapes
:         АQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :╢
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:         @:         @:         @:         @*
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         @V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         @S
mulMulSigmoid_1:y:0init_c*
T0*'
_output_shapes
:         @N
TanhTanhsplit:output:2*
T0*'
_output_shapes
:         @U
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:         @T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         @V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         @K
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:         @Y
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:         @n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :┼
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ┤
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hinit_cstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelrecurrent_kernelbias*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*c
_output_shapesQ
O: : : : :         @:         @: : :		А:	@А:А* 
_read_only_resource_inputs
 *
_stateful_parallelism( *
bodyR
while_body_24169*
condR
while_cond_24168*b
output_shapesQ
O: : : : :         @:         @: : :		А:	@А:А*
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   ╓
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         @*
element_dtype0*
num_elementsh
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:З
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         @*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ц
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:         @[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  А?`
IdentityIdentitystrided_slice_2:output:0*
T0*'
_output_shapes
:         @]

Identity_1Identitytranspose_1:y:0*
T0*+
_output_shapes
:         @X

Identity_2Identitywhile:output:4*
T0*'
_output_shapes
:         @X

Identity_3Identitywhile:output:5*
T0*'
_output_shapes
:         @I

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:                  	:         @:         @:		А:	@А:А*=
api_implements+)lstm_e5590ddf-237b-46f1-9660-fe95f9321bd6*
api_preferred_deviceCPU*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :                  	
 
_user_specified_nameinputs:OK
'
_output_shapes
:         @
 
_user_specified_nameinit_h:OK
'
_output_shapes
:         @
 
_user_specified_nameinit_c:GC

_output_shapes
:		А
 
_user_specified_namekernel:QM

_output_shapes
:	@А
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:А

_user_specified_namebias
ёK
а
&__forward_gpu_lstm_with_fallback_22065

inputs
init_h_0
init_c_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4
strided_slice
cudnnrnn

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
expanddims_1
concat_1
transpose_perm

init_h

init_c
concat_1_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
transpose_7_perm
transpose_8_perm
split_2_split_dim
split_split_dim
split_1_split_dim
concat_axisИc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          B
transpose_0	Transposeinputstranspose/perm:output:0*
T0P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : q

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0*+
_output_shapes
:         @R
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : u
ExpandDims_1
ExpandDimsinit_c_0ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:         @Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :И
splitSplitsplit/split_dim:output:0kernel*
T0*<
_output_shapes*
(:	@:	@:	@:	@*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ц
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*<
_output_shapes*
(:@@:@@:@@:@@*
	num_splitY

zeros_likeConst*
_output_shapes	
:А*
dtype0*
valueBА*    M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : r
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes	
:АS
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Э
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*D
_output_shapes2
0:@:@:@:@:@:@:@:@*
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
         a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes

:@	Y
ReshapeReshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:└a
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes

:@	[
	Reshape_1Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:└a
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

:@	[
	Reshape_2Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:└a
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes

:@	[
	Reshape_3Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:└a
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:@@[
	Reshape_4Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:А a
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0*
_output_shapes

:@@[
	Reshape_5Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:А a
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0*
_output_shapes

:@@[
	Reshape_6Reshapetranspose_7:y:0Const:output:0*
T0*
_output_shapes	
:А a
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0*
_output_shapes

:@@[
	Reshape_7Reshapetranspose_8:y:0Const:output:0*
T0*
_output_shapes	
:А [
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
:@[
	Reshape_9Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
:@\

Reshape_10Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
:@\

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
:@\

Reshape_12Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
:@\

Reshape_13Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
:@\

Reshape_14Reshapesplit_2:output:6Const:output:0*
T0*
_output_shapes
:@\

Reshape_15Reshapesplit_2:output:7Const:output:0*
T0*
_output_shapes
:@O
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Д

concat_1_0ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0╫
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1_0:output:0*
T0*f
_output_shapesT
R:                  @:         @:         @:f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
         _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╛
strided_slice_0StridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_maskp
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:         @*
squeeze_dims
 r
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*'
_output_shapes
:         @*
squeeze_dims
 R
ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B :Е
ExpandDims_2
ExpandDimsstrided_slice_0:output:0ExpandDims_2/dim:output:0*
T0*+
_output_shapes
:         @[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @`
IdentityIdentitystrided_slice_0:output:0*
T0*'
_output_shapes
:         @c

Identity_1IdentityExpandDims_2:output:0*
T0*+
_output_shapes
:         @Z

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:         @\

Identity_3IdentitySqueeze_1:output:0*
T0*'
_output_shapes
:         @I

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
concat_1concat_1_0:output:0"'
concat_1_axisconcat_1/axis:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output_h:0"!

cudnnrnn_0CudnnRNN:output_c:0"

cudnnrnn_1CudnnRNN:output:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"!

expanddimsExpandDims:output:0"%
expanddims_1ExpandDims_1:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
init_cinit_c_0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0")
strided_slicestrided_slice_0:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0"-
transpose_7_permtranspose_7/perm:output:0"-
transpose_8_permtranspose_8/perm:output:0")
transpose_permtranspose/perm:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:                  	:         @:         @:		А:	@А:А*=
api_implements+)lstm_d4300e95-cf4b-4bbb-ba38-7f8ecfb78490*
api_preferred_deviceGPU*U
backward_function_name;9__inference___backward_gpu_lstm_with_fallback_21890_22066*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :                  	
 
_user_specified_nameinputs:OK
'
_output_shapes
:         @
 
_user_specified_nameinit_h:OK
'
_output_shapes
:         @
 
_user_specified_nameinit_c:GC

_output_shapes
:		А
 
_user_specified_namekernel:QM

_output_shapes
:	@А
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:А

_user_specified_namebias
М	
╝
while_cond_21214
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice3
/while_while_cond_21214___redundant_placeholder03
/while_while_cond_21214___redundant_placeholder13
/while_while_cond_21214___redundant_placeholder23
/while_while_cond_21214___redundant_placeholder3
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :         @:         @: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         @:-)
'
_output_shapes
:         @:

_output_shapes
: :

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
╟@
╦
(__inference_gpu_lstm_with_fallback_23012

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4Иc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :                  	P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : o

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*+
_output_shapes
:         @R
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : s
ExpandDims_1
ExpandDimsinit_cExpandDims_1/dim:output:0*
T0*+
_output_shapes
:         @Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :И
splitSplitsplit/split_dim:output:0kernel*
T0*<
_output_shapes*
(:	@:	@:	@:	@*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ц
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*<
_output_shapes*
(:@@:@@:@@:@@*
	num_splitY

zeros_likeConst*
_output_shapes	
:А*
dtype0*
valueBА*    M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : r
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes	
:АS
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Э
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*D
_output_shapes2
0:@:@:@:@:@:@:@:@*
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
         a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes

:@	Y
ReshapeReshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:└a
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes

:@	[
	Reshape_1Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:└a
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

:@	[
	Reshape_2Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:└a
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes

:@	[
	Reshape_3Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:└a
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:@@[
	Reshape_4Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:А a
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0*
_output_shapes

:@@[
	Reshape_5Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:А a
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0*
_output_shapes

:@@[
	Reshape_6Reshapetranspose_7:y:0Const:output:0*
T0*
_output_shapes	
:А a
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0*
_output_shapes

:@@[
	Reshape_7Reshapetranspose_8:y:0Const:output:0*
T0*
_output_shapes	
:А [
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
:@[
	Reshape_9Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
:@\

Reshape_10Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
:@\

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
:@\

Reshape_12Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
:@\

Reshape_13Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
:@\

Reshape_14Reshapesplit_2:output:6Const:output:0*
T0*
_output_shapes
:@\

Reshape_15Reshapesplit_2:output:7Const:output:0*
T0*
_output_shapes
:@O
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : а
concat_1ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes

:АЦ╙
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1:output:0*
T0*f
_output_shapesT
R:                  @:         @:         @:f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
         _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:х
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         @*
shrink_axis_maskp
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:         @*
squeeze_dims
 r
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*'
_output_shapes
:         @*
squeeze_dims
 R
ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B :Г
ExpandDims_2
ExpandDimsstrided_slice:output:0ExpandDims_2/dim:output:0*
T0*+
_output_shapes
:         @[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @^
IdentityIdentitystrided_slice:output:0*
T0*'
_output_shapes
:         @c

Identity_1IdentityExpandDims_2:output:0*
T0*+
_output_shapes
:         @Z

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:         @\

Identity_3IdentitySqueeze_1:output:0*
T0*'
_output_shapes
:         @I

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:                  	:         @:         @:		А:	@А:А*=
api_implements+)lstm_6046d01b-8f35-430c-8827-9bde89cc14a6*
api_preferred_deviceGPU*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :                  	
 
_user_specified_nameinputs:OK
'
_output_shapes
:         @
 
_user_specified_nameinit_h:OK
'
_output_shapes
:         @
 
_user_specified_nameinit_c:GC

_output_shapes
:		А
 
_user_specified_namekernel:QM

_output_shapes
:	@А
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:А

_user_specified_namebias
ёK
а
&__forward_gpu_lstm_with_fallback_24096

inputs
init_h_0
init_c_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4
strided_slice
cudnnrnn

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
expanddims_1
concat_1
transpose_perm

init_h

init_c
concat_1_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
transpose_7_perm
transpose_8_perm
split_2_split_dim
split_split_dim
split_1_split_dim
concat_axisИc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          B
transpose_0	Transposeinputstranspose/perm:output:0*
T0P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : q

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0*+
_output_shapes
:         @R
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : u
ExpandDims_1
ExpandDimsinit_c_0ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:         @Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :И
splitSplitsplit/split_dim:output:0kernel*
T0*<
_output_shapes*
(:	@:	@:	@:	@*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ц
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*<
_output_shapes*
(:@@:@@:@@:@@*
	num_splitY

zeros_likeConst*
_output_shapes	
:А*
dtype0*
valueBА*    M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : r
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes	
:АS
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Э
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*D
_output_shapes2
0:@:@:@:@:@:@:@:@*
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
         a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes

:@	Y
ReshapeReshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:└a
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes

:@	[
	Reshape_1Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:└a
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

:@	[
	Reshape_2Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:└a
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes

:@	[
	Reshape_3Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:└a
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:@@[
	Reshape_4Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:А a
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0*
_output_shapes

:@@[
	Reshape_5Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:А a
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0*
_output_shapes

:@@[
	Reshape_6Reshapetranspose_7:y:0Const:output:0*
T0*
_output_shapes	
:А a
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0*
_output_shapes

:@@[
	Reshape_7Reshapetranspose_8:y:0Const:output:0*
T0*
_output_shapes	
:А [
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
:@[
	Reshape_9Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
:@\

Reshape_10Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
:@\

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
:@\

Reshape_12Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
:@\

Reshape_13Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
:@\

Reshape_14Reshapesplit_2:output:6Const:output:0*
T0*
_output_shapes
:@\

Reshape_15Reshapesplit_2:output:7Const:output:0*
T0*
_output_shapes
:@O
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Д

concat_1_0ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0╫
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1_0:output:0*
T0*f
_output_shapesT
R:                  @:         @:         @:f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
         _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╛
strided_slice_0StridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_maskp
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:         @*
squeeze_dims
 r
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*'
_output_shapes
:         @*
squeeze_dims
 R
ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B :Е
ExpandDims_2
ExpandDimsstrided_slice_0:output:0ExpandDims_2/dim:output:0*
T0*+
_output_shapes
:         @[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @`
IdentityIdentitystrided_slice_0:output:0*
T0*'
_output_shapes
:         @c

Identity_1IdentityExpandDims_2:output:0*
T0*+
_output_shapes
:         @Z

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:         @\

Identity_3IdentitySqueeze_1:output:0*
T0*'
_output_shapes
:         @I

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
concat_1concat_1_0:output:0"'
concat_1_axisconcat_1/axis:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output_h:0"!

cudnnrnn_0CudnnRNN:output_c:0"

cudnnrnn_1CudnnRNN:output:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"!

expanddimsExpandDims:output:0"%
expanddims_1ExpandDims_1:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
init_cinit_c_0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0")
strided_slicestrided_slice_0:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0"-
transpose_7_permtranspose_7/perm:output:0"-
transpose_8_permtranspose_8/perm:output:0")
transpose_permtranspose/perm:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:                  	:         @:         @:		А:	@А:А*=
api_implements+)lstm_5a52159d-dcb6-440d-9622-97ba434935b9*
api_preferred_deviceGPU*U
backward_function_name;9__inference___backward_gpu_lstm_with_fallback_23921_24097*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :                  	
 
_user_specified_nameinputs:OK
'
_output_shapes
:         @
 
_user_specified_nameinit_h:OK
'
_output_shapes
:         @
 
_user_specified_nameinit_c:GC

_output_shapes
:		А
 
_user_specified_namekernel:QM

_output_shapes
:	@А
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:А

_user_specified_namebias
╗;
┐
__inference_standard_lstm_24684

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :                  	B
ShapeShapetranspose:y:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ▓
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    	   р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:щ
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         	*
shrink_axis_maske
MatMulMatMulstrided_slice_1:output:0kernel*
T0*(
_output_shapes
:         А_
MatMul_1MatMulinit_hrecurrent_kernel*
T0*(
_output_shapes
:         Аe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         АT
BiasAddBiasAddadd:z:0bias*
T0*(
_output_shapes
:         АQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :╢
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:         @:         @:         @:         @*
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         @V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         @S
mulMulSigmoid_1:y:0init_c*
T0*'
_output_shapes
:         @N
TanhTanhsplit:output:2*
T0*'
_output_shapes
:         @U
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:         @T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         @V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         @K
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:         @Y
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:         @n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :┼
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ┤
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hinit_cstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelrecurrent_kernelbias*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*c
_output_shapesQ
O: : : : :         @:         @: : :		А:	@А:А* 
_read_only_resource_inputs
 *
_stateful_parallelism( *
bodyR
while_body_24598*
condR
while_cond_24597*b
output_shapesQ
O: : : : :         @:         @: : :		А:	@А:А*
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   ╓
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         @*
element_dtype0*
num_elementsh
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:З
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         @*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ц
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:         @[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  А?`
IdentityIdentitystrided_slice_2:output:0*
T0*'
_output_shapes
:         @]

Identity_1Identitytranspose_1:y:0*
T0*+
_output_shapes
:         @X

Identity_2Identitywhile:output:4*
T0*'
_output_shapes
:         @X

Identity_3Identitywhile:output:5*
T0*'
_output_shapes
:         @I

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:                  	:         @:         @:		А:	@А:А*=
api_implements+)lstm_8dc90f36-b787-4459-a772-dec1d233e41f*
api_preferred_deviceCPU*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :                  	
 
_user_specified_nameinputs:OK
'
_output_shapes
:         @
 
_user_specified_nameinit_h:OK
'
_output_shapes
:         @
 
_user_specified_nameinit_c:GC

_output_shapes
:		А
 
_user_specified_namekernel:QM

_output_shapes
:	@А
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:А

_user_specified_namebias
в
╣
?__inference_lstm_layer_call_and_return_conditional_losses_21123

inputs/
read_readvariableop_resource:		А1
read_1_readvariableop_resource:	@А-
read_2_readvariableop_resource:	А

identity_3ИвRead/ReadVariableOpвRead_1/ReadVariableOpвRead_2/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:         @R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:         @q
Read/ReadVariableOpReadVariableOpread_readvariableop_resource*
_output_shapes
:		А*
dtype0[
IdentityIdentityRead/ReadVariableOp:value:0*
T0*
_output_shapes
:		Аu
Read_1/ReadVariableOpReadVariableOpread_1_readvariableop_resource*
_output_shapes
:	@А*
dtype0_

Identity_1IdentityRead_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	@Аq
Read_2/ReadVariableOpReadVariableOpread_2_readvariableop_resource*
_output_shapes	
:А*
dtype0[

Identity_2IdentityRead_2/ReadVariableOp:value:0*
T0*
_output_shapes	
:А▒
PartitionedCallPartitionedCallinputszeros:output:0zeros_1:output:0Identity:output:0Identity_1:output:0Identity_2:output:0*
Tin

2*
Tout	
2*
_collective_manager_ids
 *f
_output_shapesT
R:         @:         @:         @:         @: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *(
f#R!
__inference_standard_lstm_20850i

Identity_3IdentityPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         @М
NoOpNoOp^Read/ReadVariableOp^Read_1/ReadVariableOp^Read_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  	: : : 2*
Read/ReadVariableOpRead/ReadVariableOp2.
Read_1/ReadVariableOpRead_1/ReadVariableOp2.
Read_2/ReadVariableOpRead_2/ReadVariableOp:\ X
4
_output_shapes"
 :                  	
 
_user_specified_nameinputs
М	
╝
while_cond_21708
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice3
/while_while_cond_21708___redundant_placeholder03
/while_while_cond_21708___redundant_placeholder13
/while_while_cond_21708___redundant_placeholder23
/while_while_cond_21708___redundant_placeholder3
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :         @:         @: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         @:-)
'
_output_shapes
:         @:

_output_shapes
: :

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
╛
Ф
%__inference_dense_layer_call_fn_24966

inputs
unknown:	@э
	unknown_0:	э
identityИвStatefulPartitionedCall╓
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         э*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_21592p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         э`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         @: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
ёK
а
&__forward_gpu_lstm_with_fallback_24954

inputs
init_h_0
init_c_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4
strided_slice
cudnnrnn

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
expanddims_1
concat_1
transpose_perm

init_h

init_c
concat_1_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
transpose_7_perm
transpose_8_perm
split_2_split_dim
split_split_dim
split_1_split_dim
concat_axisИc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          B
transpose_0	Transposeinputstranspose/perm:output:0*
T0P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : q

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0*+
_output_shapes
:         @R
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : u
ExpandDims_1
ExpandDimsinit_c_0ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:         @Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :И
splitSplitsplit/split_dim:output:0kernel*
T0*<
_output_shapes*
(:	@:	@:	@:	@*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ц
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*<
_output_shapes*
(:@@:@@:@@:@@*
	num_splitY

zeros_likeConst*
_output_shapes	
:А*
dtype0*
valueBА*    M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : r
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes	
:АS
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Э
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*D
_output_shapes2
0:@:@:@:@:@:@:@:@*
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
         a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes

:@	Y
ReshapeReshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:└a
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes

:@	[
	Reshape_1Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:└a
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

:@	[
	Reshape_2Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:└a
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes

:@	[
	Reshape_3Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:└a
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:@@[
	Reshape_4Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:А a
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0*
_output_shapes

:@@[
	Reshape_5Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:А a
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0*
_output_shapes

:@@[
	Reshape_6Reshapetranspose_7:y:0Const:output:0*
T0*
_output_shapes	
:А a
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0*
_output_shapes

:@@[
	Reshape_7Reshapetranspose_8:y:0Const:output:0*
T0*
_output_shapes	
:А [
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
:@[
	Reshape_9Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
:@\

Reshape_10Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
:@\

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
:@\

Reshape_12Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
:@\

Reshape_13Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
:@\

Reshape_14Reshapesplit_2:output:6Const:output:0*
T0*
_output_shapes
:@\

Reshape_15Reshapesplit_2:output:7Const:output:0*
T0*
_output_shapes
:@O
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Д

concat_1_0ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0╫
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1_0:output:0*
T0*f
_output_shapesT
R:                  @:         @:         @:f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
         _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╛
strided_slice_0StridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_maskp
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:         @*
squeeze_dims
 r
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*'
_output_shapes
:         @*
squeeze_dims
 R
ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B :Е
ExpandDims_2
ExpandDimsstrided_slice_0:output:0ExpandDims_2/dim:output:0*
T0*+
_output_shapes
:         @[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @`
IdentityIdentitystrided_slice_0:output:0*
T0*'
_output_shapes
:         @c

Identity_1IdentityExpandDims_2:output:0*
T0*+
_output_shapes
:         @Z

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:         @\

Identity_3IdentitySqueeze_1:output:0*
T0*'
_output_shapes
:         @I

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
concat_1concat_1_0:output:0"'
concat_1_axisconcat_1/axis:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output_h:0"!

cudnnrnn_0CudnnRNN:output_c:0"

cudnnrnn_1CudnnRNN:output:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"!

expanddimsExpandDims:output:0"%
expanddims_1ExpandDims_1:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
init_cinit_c_0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0")
strided_slicestrided_slice_0:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0"-
transpose_7_permtranspose_7/perm:output:0"-
transpose_8_permtranspose_8/perm:output:0")
transpose_permtranspose/perm:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:                  	:         @:         @:		А:	@А:А*=
api_implements+)lstm_8dc90f36-b787-4459-a772-dec1d233e41f*
api_preferred_deviceGPU*U
backward_function_name;9__inference___backward_gpu_lstm_with_fallback_24779_24955*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :                  	
 
_user_specified_nameinputs:OK
'
_output_shapes
:         @
 
_user_specified_nameinit_h:OK
'
_output_shapes
:         @
 
_user_specified_nameinit_c:GC

_output_shapes
:		А
 
_user_specified_namekernel:QM

_output_shapes
:	@А
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:А

_user_specified_namebias
╟@
╦
(__inference_gpu_lstm_with_fallback_20944

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4Иc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :                  	P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : o

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*+
_output_shapes
:         @R
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : s
ExpandDims_1
ExpandDimsinit_cExpandDims_1/dim:output:0*
T0*+
_output_shapes
:         @Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :И
splitSplitsplit/split_dim:output:0kernel*
T0*<
_output_shapes*
(:	@:	@:	@:	@*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ц
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*<
_output_shapes*
(:@@:@@:@@:@@*
	num_splitY

zeros_likeConst*
_output_shapes	
:А*
dtype0*
valueBА*    M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : r
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes	
:АS
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Э
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*D
_output_shapes2
0:@:@:@:@:@:@:@:@*
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
         a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes

:@	Y
ReshapeReshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:└a
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes

:@	[
	Reshape_1Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:└a
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

:@	[
	Reshape_2Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:└a
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes

:@	[
	Reshape_3Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:└a
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:@@[
	Reshape_4Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:А a
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0*
_output_shapes

:@@[
	Reshape_5Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:А a
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0*
_output_shapes

:@@[
	Reshape_6Reshapetranspose_7:y:0Const:output:0*
T0*
_output_shapes	
:А a
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0*
_output_shapes

:@@[
	Reshape_7Reshapetranspose_8:y:0Const:output:0*
T0*
_output_shapes	
:А [
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
:@[
	Reshape_9Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
:@\

Reshape_10Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
:@\

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
:@\

Reshape_12Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
:@\

Reshape_13Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
:@\

Reshape_14Reshapesplit_2:output:6Const:output:0*
T0*
_output_shapes
:@\

Reshape_15Reshapesplit_2:output:7Const:output:0*
T0*
_output_shapes
:@O
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : а
concat_1ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes

:АЦ╙
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1:output:0*
T0*f
_output_shapesT
R:                  @:         @:         @:f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
         _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:х
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         @*
shrink_axis_maskp
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:         @*
squeeze_dims
 r
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*'
_output_shapes
:         @*
squeeze_dims
 R
ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B :Г
ExpandDims_2
ExpandDimsstrided_slice:output:0ExpandDims_2/dim:output:0*
T0*+
_output_shapes
:         @[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @^
IdentityIdentitystrided_slice:output:0*
T0*'
_output_shapes
:         @c

Identity_1IdentityExpandDims_2:output:0*
T0*+
_output_shapes
:         @Z

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:         @\

Identity_3IdentitySqueeze_1:output:0*
T0*'
_output_shapes
:         @I

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:                  	:         @:         @:		А:	@А:А*=
api_implements+)lstm_733a4ab0-824c-43e0-83a3-926a8e9369c1*
api_preferred_deviceGPU*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :                  	
 
_user_specified_nameinputs:OK
'
_output_shapes
:         @
 
_user_specified_nameinit_h:OK
'
_output_shapes
:         @
 
_user_specified_nameinit_c:GC

_output_shapes
:		А
 
_user_specified_namekernel:QM

_output_shapes
:	@А
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:А

_user_specified_namebias
в
╣
?__inference_lstm_layer_call_and_return_conditional_losses_21574

inputs/
read_readvariableop_resource:		А1
read_1_readvariableop_resource:	@А-
read_2_readvariableop_resource:	А

identity_3ИвRead/ReadVariableOpвRead_1/ReadVariableOpвRead_2/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:         @R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:         @q
Read/ReadVariableOpReadVariableOpread_readvariableop_resource*
_output_shapes
:		А*
dtype0[
IdentityIdentityRead/ReadVariableOp:value:0*
T0*
_output_shapes
:		Аu
Read_1/ReadVariableOpReadVariableOpread_1_readvariableop_resource*
_output_shapes
:	@А*
dtype0_

Identity_1IdentityRead_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	@Аq
Read_2/ReadVariableOpReadVariableOpread_2_readvariableop_resource*
_output_shapes	
:А*
dtype0[

Identity_2IdentityRead_2/ReadVariableOp:value:0*
T0*
_output_shapes	
:А▒
PartitionedCallPartitionedCallinputszeros:output:0zeros_1:output:0Identity:output:0Identity_1:output:0Identity_2:output:0*
Tin

2*
Tout	
2*
_collective_manager_ids
 *f
_output_shapesT
R:         @:         @:         @:         @: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *(
f#R!
__inference_standard_lstm_21301i

Identity_3IdentityPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         @М
NoOpNoOp^Read/ReadVariableOp^Read_1/ReadVariableOp^Read_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  	: : : 2*
Read/ReadVariableOpRead/ReadVariableOp2.
Read_1/ReadVariableOpRead_1/ReadVariableOp2.
Read_2/ReadVariableOpRead_2/ReadVariableOp:\ X
4
_output_shapes"
 :                  	
 
_user_specified_nameinputs
З
│
$__inference_lstm_layer_call_fn_23219
inputs_0
unknown:		А
	unknown_0:	@А
	unknown_1:	А
identityИвStatefulPartitionedCallу
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *H
fCRA
?__inference_lstm_layer_call_and_return_conditional_losses_21123o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         @`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  	: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :                  	
"
_user_specified_name
inputs/0
Б
▒
$__inference_lstm_layer_call_fn_23241

inputs
unknown:		А
	unknown_0:	@А
	unknown_1:	А
identityИвStatefulPartitionedCallс
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *H
fCRA
?__inference_lstm_layer_call_and_return_conditional_losses_22068o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         @`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  	: : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :                  	
 
_user_specified_nameinputs
ёK
а
&__forward_gpu_lstm_with_fallback_23188

inputs
init_h_0
init_c_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4
strided_slice
cudnnrnn

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
expanddims_1
concat_1
transpose_perm

init_h

init_c
concat_1_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
transpose_7_perm
transpose_8_perm
split_2_split_dim
split_split_dim
split_1_split_dim
concat_axisИc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          B
transpose_0	Transposeinputstranspose/perm:output:0*
T0P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : q

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0*+
_output_shapes
:         @R
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : u
ExpandDims_1
ExpandDimsinit_c_0ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:         @Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :И
splitSplitsplit/split_dim:output:0kernel*
T0*<
_output_shapes*
(:	@:	@:	@:	@*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ц
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*<
_output_shapes*
(:@@:@@:@@:@@*
	num_splitY

zeros_likeConst*
_output_shapes	
:А*
dtype0*
valueBА*    M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : r
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes	
:АS
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Э
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*D
_output_shapes2
0:@:@:@:@:@:@:@:@*
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
         a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes

:@	Y
ReshapeReshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:└a
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes

:@	[
	Reshape_1Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:└a
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

:@	[
	Reshape_2Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:└a
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes

:@	[
	Reshape_3Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:└a
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:@@[
	Reshape_4Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:А a
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0*
_output_shapes

:@@[
	Reshape_5Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:А a
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0*
_output_shapes

:@@[
	Reshape_6Reshapetranspose_7:y:0Const:output:0*
T0*
_output_shapes	
:А a
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0*
_output_shapes

:@@[
	Reshape_7Reshapetranspose_8:y:0Const:output:0*
T0*
_output_shapes	
:А [
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
:@[
	Reshape_9Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
:@\

Reshape_10Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
:@\

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
:@\

Reshape_12Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
:@\

Reshape_13Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
:@\

Reshape_14Reshapesplit_2:output:6Const:output:0*
T0*
_output_shapes
:@\

Reshape_15Reshapesplit_2:output:7Const:output:0*
T0*
_output_shapes
:@O
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Д

concat_1_0ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0╫
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1_0:output:0*
T0*f
_output_shapesT
R:                  @:         @:         @:f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
         _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╛
strided_slice_0StridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_maskp
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:         @*
squeeze_dims
 r
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*'
_output_shapes
:         @*
squeeze_dims
 R
ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B :Е
ExpandDims_2
ExpandDimsstrided_slice_0:output:0ExpandDims_2/dim:output:0*
T0*+
_output_shapes
:         @[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @`
IdentityIdentitystrided_slice_0:output:0*
T0*'
_output_shapes
:         @c

Identity_1IdentityExpandDims_2:output:0*
T0*+
_output_shapes
:         @Z

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:         @\

Identity_3IdentitySqueeze_1:output:0*
T0*'
_output_shapes
:         @I

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
concat_1concat_1_0:output:0"'
concat_1_axisconcat_1/axis:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output_h:0"!

cudnnrnn_0CudnnRNN:output_c:0"

cudnnrnn_1CudnnRNN:output:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"!

expanddimsExpandDims:output:0"%
expanddims_1ExpandDims_1:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
init_cinit_c_0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0")
strided_slicestrided_slice_0:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0"-
transpose_7_permtranspose_7/perm:output:0"-
transpose_8_permtranspose_8/perm:output:0")
transpose_permtranspose/perm:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:                  	:         @:         @:		А:	@А:А*=
api_implements+)lstm_6046d01b-8f35-430c-8827-9bde89cc14a6*
api_preferred_deviceGPU*U
backward_function_name;9__inference___backward_gpu_lstm_with_fallback_23013_23189*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :                  	
 
_user_specified_nameinputs:OK
'
_output_shapes
:         @
 
_user_specified_nameinit_h:OK
'
_output_shapes
:         @
 
_user_specified_nameinit_c:GC

_output_shapes
:		А
 
_user_specified_namekernel:QM

_output_shapes
:	@А
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:А

_user_specified_namebias
ёK
а
&__forward_gpu_lstm_with_fallback_20680

inputs
init_h_0
init_c_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4
strided_slice
cudnnrnn

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
expanddims_1
concat_1
transpose_perm

init_h

init_c
concat_1_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
transpose_7_perm
transpose_8_perm
split_2_split_dim
split_split_dim
split_1_split_dim
concat_axisИc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          B
transpose_0	Transposeinputstranspose/perm:output:0*
T0P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : q

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0*+
_output_shapes
:         @R
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : u
ExpandDims_1
ExpandDimsinit_c_0ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:         @Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :И
splitSplitsplit/split_dim:output:0kernel*
T0*<
_output_shapes*
(:	@:	@:	@:	@*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ц
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*<
_output_shapes*
(:@@:@@:@@:@@*
	num_splitY

zeros_likeConst*
_output_shapes	
:А*
dtype0*
valueBА*    M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : r
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes	
:АS
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Э
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*D
_output_shapes2
0:@:@:@:@:@:@:@:@*
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
         a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes

:@	Y
ReshapeReshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:└a
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes

:@	[
	Reshape_1Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:└a
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

:@	[
	Reshape_2Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:└a
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes

:@	[
	Reshape_3Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:└a
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:@@[
	Reshape_4Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:А a
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0*
_output_shapes

:@@[
	Reshape_5Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:А a
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0*
_output_shapes

:@@[
	Reshape_6Reshapetranspose_7:y:0Const:output:0*
T0*
_output_shapes	
:А a
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0*
_output_shapes

:@@[
	Reshape_7Reshapetranspose_8:y:0Const:output:0*
T0*
_output_shapes	
:А [
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
:@[
	Reshape_9Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
:@\

Reshape_10Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
:@\

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
:@\

Reshape_12Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
:@\

Reshape_13Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
:@\

Reshape_14Reshapesplit_2:output:6Const:output:0*
T0*
_output_shapes
:@\

Reshape_15Reshapesplit_2:output:7Const:output:0*
T0*
_output_shapes
:@O
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Д

concat_1_0ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0╫
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1_0:output:0*
T0*f
_output_shapesT
R:                  @:         @:         @:f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
         _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╛
strided_slice_0StridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_maskp
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:         @*
squeeze_dims
 r
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*'
_output_shapes
:         @*
squeeze_dims
 R
ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B :Е
ExpandDims_2
ExpandDimsstrided_slice_0:output:0ExpandDims_2/dim:output:0*
T0*+
_output_shapes
:         @[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @`
IdentityIdentitystrided_slice_0:output:0*
T0*'
_output_shapes
:         @c

Identity_1IdentityExpandDims_2:output:0*
T0*+
_output_shapes
:         @Z

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:         @\

Identity_3IdentitySqueeze_1:output:0*
T0*'
_output_shapes
:         @I

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
concat_1concat_1_0:output:0"'
concat_1_axisconcat_1/axis:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output_h:0"!

cudnnrnn_0CudnnRNN:output_c:0"

cudnnrnn_1CudnnRNN:output:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"!

expanddimsExpandDims:output:0"%
expanddims_1ExpandDims_1:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
init_cinit_c_0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0")
strided_slicestrided_slice_0:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0"-
transpose_7_permtranspose_7/perm:output:0"-
transpose_8_permtranspose_8/perm:output:0")
transpose_permtranspose/perm:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:                  	:         @:         @:		А:	@А:А*=
api_implements+)lstm_8d97cef0-5325-4590-9604-659863d02e30*
api_preferred_deviceGPU*U
backward_function_name;9__inference___backward_gpu_lstm_with_fallback_20505_20681*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :                  	
 
_user_specified_nameinputs:OK
'
_output_shapes
:         @
 
_user_specified_nameinit_h:OK
'
_output_shapes
:         @
 
_user_specified_nameinit_c:GC

_output_shapes
:		А
 
_user_specified_namekernel:QM

_output_shapes
:	@А
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:А

_user_specified_namebias
╗;
┐
__inference_standard_lstm_20410

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :                  	B
ShapeShapetranspose:y:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ▓
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    	   р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:щ
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         	*
shrink_axis_maske
MatMulMatMulstrided_slice_1:output:0kernel*
T0*(
_output_shapes
:         А_
MatMul_1MatMulinit_hrecurrent_kernel*
T0*(
_output_shapes
:         Аe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         АT
BiasAddBiasAddadd:z:0bias*
T0*(
_output_shapes
:         АQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :╢
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:         @:         @:         @:         @*
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         @V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         @S
mulMulSigmoid_1:y:0init_c*
T0*'
_output_shapes
:         @N
TanhTanhsplit:output:2*
T0*'
_output_shapes
:         @U
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:         @T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         @V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         @K
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:         @Y
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:         @n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :┼
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ┤
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hinit_cstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelrecurrent_kernelbias*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*c
_output_shapesQ
O: : : : :         @:         @: : :		А:	@А:А* 
_read_only_resource_inputs
 *
_stateful_parallelism( *
bodyR
while_body_20324*
condR
while_cond_20323*b
output_shapesQ
O: : : : :         @:         @: : :		А:	@А:А*
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   ╓
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         @*
element_dtype0*
num_elementsh
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:З
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         @*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ц
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:         @[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  А?`
IdentityIdentitystrided_slice_2:output:0*
T0*'
_output_shapes
:         @]

Identity_1Identitytranspose_1:y:0*
T0*+
_output_shapes
:         @X

Identity_2Identitywhile:output:4*
T0*'
_output_shapes
:         @X

Identity_3Identitywhile:output:5*
T0*'
_output_shapes
:         @I

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:                  	:         @:         @:		А:	@А:А*=
api_implements+)lstm_8d97cef0-5325-4590-9604-659863d02e30*
api_preferred_deviceCPU*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :                  	
 
_user_specified_nameinputs:OK
'
_output_shapes
:         @
 
_user_specified_nameinit_h:OK
'
_output_shapes
:         @
 
_user_specified_nameinit_c:GC

_output_shapes
:		А
 
_user_specified_namekernel:QM

_output_shapes
:	@А
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:А

_user_specified_namebias
╩	
є
@__inference_dense_layer_call_and_return_conditional_losses_24976

inputs1
matmul_readvariableop_resource:	@э.
biasadd_readvariableop_resource:	э
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@э*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         эs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:э*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         э`
IdentityIdentityBiasAdd:output:0^NoOp*
T0*(
_output_shapes
:         эw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         @: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
╟@
╦
(__inference_gpu_lstm_with_fallback_21395

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4Иc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :                  	P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : o

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*+
_output_shapes
:         @R
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : s
ExpandDims_1
ExpandDimsinit_cExpandDims_1/dim:output:0*
T0*+
_output_shapes
:         @Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :И
splitSplitsplit/split_dim:output:0kernel*
T0*<
_output_shapes*
(:	@:	@:	@:	@*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ц
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*<
_output_shapes*
(:@@:@@:@@:@@*
	num_splitY

zeros_likeConst*
_output_shapes	
:А*
dtype0*
valueBА*    M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : r
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes	
:АS
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Э
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*D
_output_shapes2
0:@:@:@:@:@:@:@:@*
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
         a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes

:@	Y
ReshapeReshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:└a
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes

:@	[
	Reshape_1Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:└a
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

:@	[
	Reshape_2Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:└a
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes

:@	[
	Reshape_3Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:└a
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:@@[
	Reshape_4Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:А a
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0*
_output_shapes

:@@[
	Reshape_5Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:А a
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0*
_output_shapes

:@@[
	Reshape_6Reshapetranspose_7:y:0Const:output:0*
T0*
_output_shapes	
:А a
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0*
_output_shapes

:@@[
	Reshape_7Reshapetranspose_8:y:0Const:output:0*
T0*
_output_shapes	
:А [
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
:@[
	Reshape_9Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
:@\

Reshape_10Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
:@\

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
:@\

Reshape_12Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
:@\

Reshape_13Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
:@\

Reshape_14Reshapesplit_2:output:6Const:output:0*
T0*
_output_shapes
:@\

Reshape_15Reshapesplit_2:output:7Const:output:0*
T0*
_output_shapes
:@O
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : а
concat_1ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes

:АЦ╙
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1:output:0*
T0*f
_output_shapesT
R:                  @:         @:         @:f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
         _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:х
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         @*
shrink_axis_maskp
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:         @*
squeeze_dims
 r
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*'
_output_shapes
:         @*
squeeze_dims
 R
ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B :Г
ExpandDims_2
ExpandDimsstrided_slice:output:0ExpandDims_2/dim:output:0*
T0*+
_output_shapes
:         @[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @^
IdentityIdentitystrided_slice:output:0*
T0*'
_output_shapes
:         @c

Identity_1IdentityExpandDims_2:output:0*
T0*+
_output_shapes
:         @Z

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:         @\

Identity_3IdentitySqueeze_1:output:0*
T0*'
_output_shapes
:         @I

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:                  	:         @:         @:		А:	@А:А*=
api_implements+)lstm_e8470502-1520-4f1d-96c5-a71b9a8e2c85*
api_preferred_deviceGPU*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :                  	
 
_user_specified_nameinputs:OK
'
_output_shapes
:         @
 
_user_specified_nameinit_h:OK
'
_output_shapes
:         @
 
_user_specified_nameinit_c:GC

_output_shapes
:		А
 
_user_specified_namekernel:QM

_output_shapes
:	@А
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:А

_user_specified_namebias
З
│
$__inference_lstm_layer_call_fn_23208
inputs_0
unknown:		А
	unknown_0:	@А
	unknown_1:	А
identityИвStatefulPartitionedCallу
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *H
fCRA
?__inference_lstm_layer_call_and_return_conditional_losses_20683o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         @`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  	: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :                  	
"
_user_specified_name
inputs/0
М	
╝
while_cond_24597
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice3
/while_while_cond_24597___redundant_placeholder03
/while_while_cond_24597___redundant_placeholder13
/while_while_cond_24597___redundant_placeholder23
/while_while_cond_24597___redundant_placeholder3
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :         @:         @: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         @:-)
'
_output_shapes
:         @:

_output_shapes
: :

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
╪)
╬
while_body_22390
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_bias_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel#
while_matmul_1_recurrent_kernel
while_biasadd_biasИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    	   ж
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         	*
element_dtype0Т
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*(
_output_shapes
:         АГ
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*(
_output_shapes
:         Аw
	while/addAddV2while/MatMul:product:0while/MatMul_1:product:0*
T0*(
_output_shapes
:         Аp
while/BiasAddBiasAddwhile/add:z:0while_biasadd_bias_0*
T0*(
_output_shapes
:         АW
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :╚
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*`
_output_shapesN
L:         @:         @:         @:         @*
	num_split`
while/SigmoidSigmoidwhile/split:output:0*
T0*'
_output_shapes
:         @b
while/Sigmoid_1Sigmoidwhile/split:output:1*
T0*'
_output_shapes
:         @l
	while/mulMulwhile/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:         @Z

while/TanhTanhwhile/split:output:2*
T0*'
_output_shapes
:         @g
while/mul_1Mulwhile/Sigmoid:y:0while/Tanh:y:0*
T0*'
_output_shapes
:         @f
while/add_1AddV2while/mul:z:0while/mul_1:z:0*
T0*'
_output_shapes
:         @b
while/Sigmoid_2Sigmoidwhile/split:output:3*
T0*'
_output_shapes
:         @W
while/Tanh_1Tanhwhile/add_1:z:0*
T0*'
_output_shapes
:         @k
while/mul_2Mulwhile/Sigmoid_2:y:0while/Tanh_1:y:0*
T0*'
_output_shapes
:         @r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : р
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/mul_2:z:0*
_output_shapes
: *
element_dtype0:щш╥O
while/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :`
while/add_2AddV2while_placeholderwhile/add_2/y:output:0*
T0*
_output_shapes
: O
while/add_3/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_3AddV2while_while_loop_counterwhile/add_3/y:output:0*
T0*
_output_shapes
: L
while/IdentityIdentitywhile/add_3:z:0*
T0*
_output_shapes
: ]
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: N
while/Identity_2Identitywhile/add_2:z:0*
T0*
_output_shapes
: y
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: _
while/Identity_4Identitywhile/mul_2:z:0*
T0*'
_output_shapes
:         @_
while/Identity_5Identitywhile/add_1:z:0*
T0*'
_output_shapes
:         @"*
while_biasadd_biaswhile_biasadd_bias_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O: : : : :         @:         @: : :		А:	@А:А: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         @:-)
'
_output_shapes
:         @:

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:		А:%	!

_output_shapes
:	@А:!


_output_shapes	
:А
М	
╝
while_cond_23310
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice3
/while_while_cond_23310___redundant_placeholder03
/while_while_cond_23310___redundant_placeholder13
/while_while_cond_23310___redundant_placeholder23
/while_while_cond_23310___redundant_placeholder3
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :         @:         @: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         @:-)
'
_output_shapes
:         @:

_output_shapes
: :

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
╪'
┴
__inference_adapt_step_22275
iterator%
add_readvariableop_resource:	 %
readvariableop_resource:	'
readvariableop_2_resource:	ИвAssignVariableOpвAssignVariableOp_1вAssignVariableOp_2вIteratorGetNextвReadVariableOpвReadVariableOp_1вReadVariableOp_2вadd/ReadVariableOp╗
IteratorGetNextIteratorGetNextiterator*
_class
loc:@iterator*,
_output_shapes
:         ┌	*+
output_shapes
:         ┌	*
output_types
2o
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       Щ
moments/meanMeanIteratorGetNext:components:0'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:	*
	keep_dims(h
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:	в
moments/SquaredDifferenceSquaredDifferenceIteratorGetNext:components:0moments/StopGradient:output:0*
T0*,
_output_shapes
:         ┌	s
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       в
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:	*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:	*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:	*
squeeze_dims
 a
ShapeShapeIteratorGetNext:components:0*
T0*
_output_shapes
:*
out_type0	a
GatherV2/indicesConst*
_output_shapes
:*
dtype0*
valueB"       O
GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Я
GatherV2GatherV2Shape:output:0GatherV2/indices:output:0GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0	*
_output_shapes
:O
ConstConst*
_output_shapes
:*
dtype0*
valueB: P
ProdProdGatherV2:output:0Const:output:0*
T0	*
_output_shapes
: f
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
: *
dtype0	X
addAddV2Prod:output:0add/ReadVariableOp:value:0*
T0	*
_output_shapes
: K
CastCastProd:output:0*

DstT0*

SrcT0	*
_output_shapes
: G
Cast_1Castadd:z:0*

DstT0*

SrcT0	*
_output_shapes
: I
truedivRealDivCast:y:0
Cast_1:y:0*
T0*
_output_shapes
: J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?H
subSubsub/x:output:0truediv:z:0*
T0*
_output_shapes
: b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	*
dtype0P
mulMulReadVariableOp:value:0sub:z:0*
T0*
_output_shapes
:	X
mul_1Mulmoments/Squeeze:output:0truediv:z:0*
T0*
_output_shapes
:	G
add_1AddV2mul:z:0	mul_1:z:0*
T0*
_output_shapes
:	d
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes
:	*
dtype0V
sub_1SubReadVariableOp_1:value:0	add_1:z:0*
T0*
_output_shapes
:	J
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @J
powPow	sub_1:z:0pow/y:output:0*
T0*
_output_shapes
:	f
ReadVariableOp_2ReadVariableOpreadvariableop_2_resource*
_output_shapes
:	*
dtype0V
add_2AddV2ReadVariableOp_2:value:0pow:z:0*
T0*
_output_shapes
:	E
mul_2Mul	add_2:z:0sub:z:0*
T0*
_output_shapes
:	V
sub_2Submoments/Squeeze:output:0	add_1:z:0*
T0*
_output_shapes
:	L
pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @N
pow_1Pow	sub_2:z:0pow_1/y:output:0*
T0*
_output_shapes
:	Z
add_3AddV2moments/Squeeze_1:output:0	pow_1:z:0*
T0*
_output_shapes
:	I
mul_3Mul	add_3:z:0truediv:z:0*
T0*
_output_shapes
:	I
add_4AddV2	mul_2:z:0	mul_3:z:0*
T0*
_output_shapes
:	е
AssignVariableOpAssignVariableOpreadvariableop_resource	add_1:z:0^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(Ш
AssignVariableOp_1AssignVariableOpreadvariableop_2_resource	add_4:z:0^ReadVariableOp_2*
_output_shapes
 *
dtype0*
validate_shape(Ъ
AssignVariableOp_2AssignVariableOpadd_readvariableop_resourceadd:z:0^add/ReadVariableOp*
_output_shapes
 *
dtype0	*
validate_shape(*(
_construction_contextkEagerRuntime*
_input_shapes

: : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22"
IteratorGetNextIteratorGetNext2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22(
add/ReadVariableOpadd/ReadVariableOp:( $
"
_user_specified_name
iterator
Г┬
у
9__inference___backward_gpu_lstm_with_fallback_23013_23189
placeholder
placeholder_1
placeholder_2
placeholder_3
placeholder_43
/gradients_expanddims_2_grad_shape_strided_slice)
%gradients_squeeze_grad_shape_cudnnrnn+
'gradients_squeeze_1_grad_shape_cudnnrnn/
+gradients_strided_slice_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9
5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_15
1gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h,
(gradients_expanddims_1_grad_shape_init_c-
)gradients_concat_1_grad_mod_concat_1_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_permA
=gradients_transpose_7_grad_invertpermutation_transpose_7_permA
=gradients_transpose_8_grad_invertpermutation_transpose_8_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim)
%gradients_concat_grad_mod_concat_axis
identity

identity_1

identity_2

identity_3

identity_4

identity_5И^
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:         @d
gradients/grad_ys_1Identityplaceholder_1*
T0*+
_output_shapes
:         @`
gradients/grad_ys_2Identityplaceholder_2*
T0*'
_output_shapes
:         @`
gradients/grad_ys_3Identityplaceholder_3*
T0*'
_output_shapes
:         @O
gradients/grad_ys_4Identityplaceholder_4*
T0*
_output_shapes
: А
!gradients/ExpandDims_2_grad/ShapeShape/gradients_expanddims_2_grad_shape_strided_slice*
T0*
_output_shapes
:к
#gradients/ExpandDims_2_grad/ReshapeReshapegradients/grad_ys_1:output:0*gradients/ExpandDims_2_grad/Shape:output:0*
T0*'
_output_shapes
:         @q
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
:д
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*+
_output_shapes
:         @u
gradients/Squeeze_1_grad/ShapeShape'gradients_squeeze_1_grad_shape_cudnnrnn*
T0*
_output_shapes
:и
 gradients/Squeeze_1_grad/ReshapeReshapegradients/grad_ys_3:output:0'gradients/Squeeze_1_grad/Shape:output:0*
T0*+
_output_shapes
:         @┼
gradients/AddNAddNgradients/grad_ys_0:output:0,gradients/ExpandDims_2_grad/Reshape:output:0*
N*
T0*&
_class
loc:@gradients/grad_ys_0*'
_output_shapes
:         @}
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:Ж
3gradients/strided_slice_grad/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
         {
1gradients/strided_slice_grad/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 
5gradients/strided_slice_grad/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:в
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0<gradients/strided_slice_grad/StridedSliceGrad/begin:output:0:gradients/strided_slice_grad/StridedSliceGrad/end:output:0>gradients/strided_slice_grad/StridedSliceGrad/strides:output:0gradients/AddN:sum:0*
Index0*
T0*4
_output_shapes"
 :                  @*
shrink_axis_maskc
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
:м
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_11gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnn'gradients_squeeze_1_grad_shape_cudnnrnn6gradients/strided_slice_grad/StridedSliceGrad:output:0'gradients/Squeeze_grad/Reshape:output:0)gradients/Squeeze_1_grad/Reshape:output:0gradients_zeros_like_cudnnrnn*
T0*j
_output_shapesX
V:                  	:         @:         @:АЦЦ
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:┘
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*4
_output_shapes"
 :                  	u
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:┼
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*'
_output_shapes
:         @y
!gradients/ExpandDims_1_grad/ShapeShape(gradients_expanddims_1_grad_shape_init_c*
T0*
_output_shapes
:╔
#gradients/ExpandDims_1_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_c_backprop:0*gradients/ExpandDims_1_grad/Shape:output:0*
T0*'
_output_shapes
:         @^
gradients/concat_1_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :Ъ
gradients/concat_1_grad/modFloorMod)gradients_concat_1_grad_mod_concat_1_axis%gradients/concat_1_grad/Rank:output:0*
T0*
_output_shapes
: h
gradients/concat_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:└j
gradients/concat_1_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:└j
gradients/concat_1_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:└j
gradients/concat_1_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:└j
gradients/concat_1_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:А j
gradients/concat_1_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:А j
gradients/concat_1_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB:А j
gradients/concat_1_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB:А i
gradients/concat_1_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB:@i
gradients/concat_1_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB:@j
 gradients/concat_1_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB:@j
 gradients/concat_1_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB:@j
 gradients/concat_1_grad/Shape_12Const*
_output_shapes
:*
dtype0*
valueB:@j
 gradients/concat_1_grad/Shape_13Const*
_output_shapes
:*
dtype0*
valueB:@j
 gradients/concat_1_grad/Shape_14Const*
_output_shapes
:*
dtype0*
valueB:@j
 gradients/concat_1_grad/Shape_15Const*
_output_shapes
:*
dtype0*
valueB:@°
$gradients/concat_1_grad/ConcatOffsetConcatOffsetgradients/concat_1_grad/mod:z:0&gradients/concat_1_grad/Shape:output:0(gradients/concat_1_grad/Shape_1:output:0(gradients/concat_1_grad/Shape_2:output:0(gradients/concat_1_grad/Shape_3:output:0(gradients/concat_1_grad/Shape_4:output:0(gradients/concat_1_grad/Shape_5:output:0(gradients/concat_1_grad/Shape_6:output:0(gradients/concat_1_grad/Shape_7:output:0(gradients/concat_1_grad/Shape_8:output:0(gradients/concat_1_grad/Shape_9:output:0)gradients/concat_1_grad/Shape_10:output:0)gradients/concat_1_grad/Shape_11:output:0)gradients/concat_1_grad/Shape_12:output:0)gradients/concat_1_grad/Shape_13:output:0)gradients/concat_1_grad/Shape_14:output:0)gradients/concat_1_grad/Shape_15:output:0*
N*t
_output_shapesb
`::::::::::::::::ь
gradients/concat_1_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:0&gradients/concat_1_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:└Ё
gradients/concat_1_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:1(gradients/concat_1_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:└Ё
gradients/concat_1_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:2(gradients/concat_1_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:└Ё
gradients/concat_1_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:3(gradients/concat_1_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:└Ё
gradients/concat_1_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:4(gradients/concat_1_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:А Ё
gradients/concat_1_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:5(gradients/concat_1_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:А Ё
gradients/concat_1_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:6(gradients/concat_1_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes	
:А Ё
gradients/concat_1_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:7(gradients/concat_1_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes	
:А я
gradients/concat_1_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:8(gradients/concat_1_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
:@я
gradients/concat_1_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:9(gradients/concat_1_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
:@Є
 gradients/concat_1_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:10)gradients/concat_1_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
:@Є
 gradients/concat_1_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:11)gradients/concat_1_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
:@Є
 gradients/concat_1_grad/Slice_12Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:12)gradients/concat_1_grad/Shape_12:output:0*
Index0*
T0*
_output_shapes
:@Є
 gradients/concat_1_grad/Slice_13Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:13)gradients/concat_1_grad/Shape_13:output:0*
Index0*
T0*
_output_shapes
:@Є
 gradients/concat_1_grad/Slice_14Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:14)gradients/concat_1_grad/Shape_14:output:0*
Index0*
T0*
_output_shapes
:@Є
 gradients/concat_1_grad/Slice_15Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:15)gradients/concat_1_grad/Shape_15:output:0*
Index0*
T0*
_output_shapes
:@m
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"@   	   б
gradients/Reshape_grad/ReshapeReshape&gradients/concat_1_grad/Slice:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes

:@	o
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"@   	   з
 gradients/Reshape_1_grad/ReshapeReshape(gradients/concat_1_grad/Slice_1:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes

:@	o
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"@   	   з
 gradients/Reshape_2_grad/ReshapeReshape(gradients/concat_1_grad/Slice_2:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes

:@	o
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"@   	   з
 gradients/Reshape_3_grad/ReshapeReshape(gradients/concat_1_grad/Slice_3:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes

:@	o
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"@   @   з
 gradients/Reshape_4_grad/ReshapeReshape(gradients/concat_1_grad/Slice_4:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:@@o
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"@   @   з
 gradients/Reshape_5_grad/ReshapeReshape(gradients/concat_1_grad/Slice_5:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:@@o
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"@   @   з
 gradients/Reshape_6_grad/ReshapeReshape(gradients/concat_1_grad/Slice_6:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:@@o
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"@   @   з
 gradients/Reshape_7_grad/ReshapeReshape(gradients/concat_1_grad/Slice_7:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes

:@@h
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:@г
 gradients/Reshape_8_grad/ReshapeReshape(gradients/concat_1_grad/Slice_8:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
:@h
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:@г
 gradients/Reshape_9_grad/ReshapeReshape(gradients/concat_1_grad/Slice_9:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
:@i
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:@ж
!gradients/Reshape_10_grad/ReshapeReshape)gradients/concat_1_grad/Slice_10:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
:@i
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:@ж
!gradients/Reshape_11_grad/ReshapeReshape)gradients/concat_1_grad/Slice_11:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
:@i
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:@ж
!gradients/Reshape_12_grad/ReshapeReshape)gradients/concat_1_grad/Slice_12:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
:@i
gradients/Reshape_13_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:@ж
!gradients/Reshape_13_grad/ReshapeReshape)gradients/concat_1_grad/Slice_13:output:0(gradients/Reshape_13_grad/Shape:output:0*
T0*
_output_shapes
:@i
gradients/Reshape_14_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:@ж
!gradients/Reshape_14_grad/ReshapeReshape)gradients/concat_1_grad/Slice_14:output:0(gradients/Reshape_14_grad/Shape:output:0*
T0*
_output_shapes
:@i
gradients/Reshape_15_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:@ж
!gradients/Reshape_15_grad/ReshapeReshape)gradients/concat_1_grad/Slice_15:output:0(gradients/Reshape_15_grad/Shape:output:0*
T0*
_output_shapes
:@Ь
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:╡
$gradients/transpose_1_grad/transpose	Transpose'gradients/Reshape_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes

:	@Ь
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:╖
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes

:	@Ь
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:╖
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes

:	@Ь
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:╖
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes

:	@Ь
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:╖
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@@Ь
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:╖
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@@Ь
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:╖
$gradients/transpose_7_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@@Ь
,gradients/transpose_8_grad/InvertPermutationInvertPermutation=gradients_transpose_8_grad_invertpermutation_transpose_8_perm*
_output_shapes
:╖
$gradients/transpose_8_grad/transpose	Transpose)gradients/Reshape_7_grad/Reshape:output:00gradients/transpose_8_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@@ч
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0*gradients/Reshape_13_grad/Reshape:output:0*gradients/Reshape_14_grad/Reshape:output:0*gradients/Reshape_15_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:Ап
gradients/split_grad/concatConcatV2(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0*
_output_shapes
:		А╡
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0(gradients/transpose_7_grad/transpose:y:0(gradients/transpose_8_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0*
_output_shapes
:	@А\
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :Т
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: f
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:Аh
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:А╩
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0*
N* 
_output_shapes
::╥
gradients/concat_grad/SliceSlice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:А╓
gradients/concat_grad/Slice_1Slice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:А{
IdentityIdentity&gradients/transpose_grad/transpose:y:0*
T0*4
_output_shapes"
 :                  	t

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0*
T0*'
_output_shapes
:         @v

Identity_2Identity,gradients/ExpandDims_1_grad/Reshape:output:0*
T0*'
_output_shapes
:         @f

Identity_3Identity$gradients/split_grad/concat:output:0*
T0*
_output_shapes
:		Аh

Identity_4Identity&gradients/split_1_grad/concat:output:0*
T0*
_output_shapes
:	@Аd

Identity_5Identity&gradients/concat_grad/Slice_1:output:0*
T0*
_output_shapes	
:А"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*(
_construction_contextkEagerRuntime*И
_input_shapesЎ
є:         @:         @:         @:         @: :         @:         @:         @:                  @::                  	:         @:         @:АЦ::         @:         @: ::::::::: : : : *=
api_implements+)lstm_6046d01b-8f35-430c-8827-9bde89cc14a6*
api_preferred_deviceGPU*A
forward_function_name(&__forward_gpu_lstm_with_fallback_23188*
go_backwards( *

time_major( :- )
'
_output_shapes
:         @:1-
+
_output_shapes
:         @:-)
'
_output_shapes
:         @:-)
'
_output_shapes
:         @:

_output_shapes
: :-)
'
_output_shapes
:         @:1-
+
_output_shapes
:         @:1-
+
_output_shapes
:         @::6
4
_output_shapes"
 :                  @:	

_output_shapes
:::
6
4
_output_shapes"
 :                  	:1-
+
_output_shapes
:         @:1-
+
_output_shapes
:         @:"

_output_shapes

:АЦ: 

_output_shapes
::-)
'
_output_shapes
:         @:-)
'
_output_shapes
:         @:

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ёK
а
&__forward_gpu_lstm_with_fallback_21571

inputs
init_h_0
init_c_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4
strided_slice
cudnnrnn

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
expanddims_1
concat_1
transpose_perm

init_h

init_c
concat_1_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
transpose_7_perm
transpose_8_perm
split_2_split_dim
split_split_dim
split_1_split_dim
concat_axisИc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          B
transpose_0	Transposeinputstranspose/perm:output:0*
T0P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : q

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0*+
_output_shapes
:         @R
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : u
ExpandDims_1
ExpandDimsinit_c_0ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:         @Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :И
splitSplitsplit/split_dim:output:0kernel*
T0*<
_output_shapes*
(:	@:	@:	@:	@*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ц
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*<
_output_shapes*
(:@@:@@:@@:@@*
	num_splitY

zeros_likeConst*
_output_shapes	
:А*
dtype0*
valueBА*    M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : r
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes	
:АS
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Э
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*D
_output_shapes2
0:@:@:@:@:@:@:@:@*
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
         a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes

:@	Y
ReshapeReshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:└a
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes

:@	[
	Reshape_1Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:└a
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

:@	[
	Reshape_2Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:└a
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes

:@	[
	Reshape_3Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:└a
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:@@[
	Reshape_4Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:А a
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0*
_output_shapes

:@@[
	Reshape_5Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:А a
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0*
_output_shapes

:@@[
	Reshape_6Reshapetranspose_7:y:0Const:output:0*
T0*
_output_shapes	
:А a
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0*
_output_shapes

:@@[
	Reshape_7Reshapetranspose_8:y:0Const:output:0*
T0*
_output_shapes	
:А [
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
:@[
	Reshape_9Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
:@\

Reshape_10Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
:@\

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
:@\

Reshape_12Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
:@\

Reshape_13Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
:@\

Reshape_14Reshapesplit_2:output:6Const:output:0*
T0*
_output_shapes
:@\

Reshape_15Reshapesplit_2:output:7Const:output:0*
T0*
_output_shapes
:@O
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Д

concat_1_0ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0╫
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1_0:output:0*
T0*f
_output_shapesT
R:                  @:         @:         @:f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
         _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╛
strided_slice_0StridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_maskp
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:         @*
squeeze_dims
 r
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*'
_output_shapes
:         @*
squeeze_dims
 R
ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B :Е
ExpandDims_2
ExpandDimsstrided_slice_0:output:0ExpandDims_2/dim:output:0*
T0*+
_output_shapes
:         @[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @`
IdentityIdentitystrided_slice_0:output:0*
T0*'
_output_shapes
:         @c

Identity_1IdentityExpandDims_2:output:0*
T0*+
_output_shapes
:         @Z

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:         @\

Identity_3IdentitySqueeze_1:output:0*
T0*'
_output_shapes
:         @I

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
concat_1concat_1_0:output:0"'
concat_1_axisconcat_1/axis:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output_h:0"!

cudnnrnn_0CudnnRNN:output_c:0"

cudnnrnn_1CudnnRNN:output:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"!

expanddimsExpandDims:output:0"%
expanddims_1ExpandDims_1:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
init_cinit_c_0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0")
strided_slicestrided_slice_0:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0"-
transpose_7_permtranspose_7/perm:output:0"-
transpose_8_permtranspose_8/perm:output:0")
transpose_permtranspose/perm:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:                  	:         @:         @:		А:	@А:А*=
api_implements+)lstm_e8470502-1520-4f1d-96c5-a71b9a8e2c85*
api_preferred_deviceGPU*U
backward_function_name;9__inference___backward_gpu_lstm_with_fallback_21396_21572*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :                  	
 
_user_specified_nameinputs:OK
'
_output_shapes
:         @
 
_user_specified_nameinit_h:OK
'
_output_shapes
:         @
 
_user_specified_nameinit_c:GC

_output_shapes
:		А
 
_user_specified_namekernel:QM

_output_shapes
:	@А
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:А

_user_specified_namebias
╪)
╬
while_body_23740
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_bias_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel#
while_matmul_1_recurrent_kernel
while_biasadd_biasИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    	   ж
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         	*
element_dtype0Т
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*(
_output_shapes
:         АГ
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*(
_output_shapes
:         Аw
	while/addAddV2while/MatMul:product:0while/MatMul_1:product:0*
T0*(
_output_shapes
:         Аp
while/BiasAddBiasAddwhile/add:z:0while_biasadd_bias_0*
T0*(
_output_shapes
:         АW
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :╚
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*`
_output_shapesN
L:         @:         @:         @:         @*
	num_split`
while/SigmoidSigmoidwhile/split:output:0*
T0*'
_output_shapes
:         @b
while/Sigmoid_1Sigmoidwhile/split:output:1*
T0*'
_output_shapes
:         @l
	while/mulMulwhile/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:         @Z

while/TanhTanhwhile/split:output:2*
T0*'
_output_shapes
:         @g
while/mul_1Mulwhile/Sigmoid:y:0while/Tanh:y:0*
T0*'
_output_shapes
:         @f
while/add_1AddV2while/mul:z:0while/mul_1:z:0*
T0*'
_output_shapes
:         @b
while/Sigmoid_2Sigmoidwhile/split:output:3*
T0*'
_output_shapes
:         @W
while/Tanh_1Tanhwhile/add_1:z:0*
T0*'
_output_shapes
:         @k
while/mul_2Mulwhile/Sigmoid_2:y:0while/Tanh_1:y:0*
T0*'
_output_shapes
:         @r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : р
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/mul_2:z:0*
_output_shapes
: *
element_dtype0:щш╥O
while/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :`
while/add_2AddV2while_placeholderwhile/add_2/y:output:0*
T0*
_output_shapes
: O
while/add_3/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_3AddV2while_while_loop_counterwhile/add_3/y:output:0*
T0*
_output_shapes
: L
while/IdentityIdentitywhile/add_3:z:0*
T0*
_output_shapes
: ]
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: N
while/Identity_2Identitywhile/add_2:z:0*
T0*
_output_shapes
: y
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: _
while/Identity_4Identitywhile/mul_2:z:0*
T0*'
_output_shapes
:         @_
while/Identity_5Identitywhile/add_1:z:0*
T0*'
_output_shapes
:         @"*
while_biasadd_biaswhile_biasadd_bias_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O: : : : :         @:         @: : :		А:	@А:А: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         @:-)
'
_output_shapes
:         @:

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:		А:%	!

_output_shapes
:	@А:!


_output_shapes	
:А
Г┬
у
9__inference___backward_gpu_lstm_with_fallback_24779_24955
placeholder
placeholder_1
placeholder_2
placeholder_3
placeholder_43
/gradients_expanddims_2_grad_shape_strided_slice)
%gradients_squeeze_grad_shape_cudnnrnn+
'gradients_squeeze_1_grad_shape_cudnnrnn/
+gradients_strided_slice_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9
5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_15
1gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h,
(gradients_expanddims_1_grad_shape_init_c-
)gradients_concat_1_grad_mod_concat_1_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_permA
=gradients_transpose_7_grad_invertpermutation_transpose_7_permA
=gradients_transpose_8_grad_invertpermutation_transpose_8_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim)
%gradients_concat_grad_mod_concat_axis
identity

identity_1

identity_2

identity_3

identity_4

identity_5И^
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:         @d
gradients/grad_ys_1Identityplaceholder_1*
T0*+
_output_shapes
:         @`
gradients/grad_ys_2Identityplaceholder_2*
T0*'
_output_shapes
:         @`
gradients/grad_ys_3Identityplaceholder_3*
T0*'
_output_shapes
:         @O
gradients/grad_ys_4Identityplaceholder_4*
T0*
_output_shapes
: А
!gradients/ExpandDims_2_grad/ShapeShape/gradients_expanddims_2_grad_shape_strided_slice*
T0*
_output_shapes
:к
#gradients/ExpandDims_2_grad/ReshapeReshapegradients/grad_ys_1:output:0*gradients/ExpandDims_2_grad/Shape:output:0*
T0*'
_output_shapes
:         @q
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
:д
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*+
_output_shapes
:         @u
gradients/Squeeze_1_grad/ShapeShape'gradients_squeeze_1_grad_shape_cudnnrnn*
T0*
_output_shapes
:и
 gradients/Squeeze_1_grad/ReshapeReshapegradients/grad_ys_3:output:0'gradients/Squeeze_1_grad/Shape:output:0*
T0*+
_output_shapes
:         @┼
gradients/AddNAddNgradients/grad_ys_0:output:0,gradients/ExpandDims_2_grad/Reshape:output:0*
N*
T0*&
_class
loc:@gradients/grad_ys_0*'
_output_shapes
:         @}
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:Ж
3gradients/strided_slice_grad/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
         {
1gradients/strided_slice_grad/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 
5gradients/strided_slice_grad/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:в
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0<gradients/strided_slice_grad/StridedSliceGrad/begin:output:0:gradients/strided_slice_grad/StridedSliceGrad/end:output:0>gradients/strided_slice_grad/StridedSliceGrad/strides:output:0gradients/AddN:sum:0*
Index0*
T0*4
_output_shapes"
 :                  @*
shrink_axis_maskc
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
:м
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_11gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnn'gradients_squeeze_1_grad_shape_cudnnrnn6gradients/strided_slice_grad/StridedSliceGrad:output:0'gradients/Squeeze_grad/Reshape:output:0)gradients/Squeeze_1_grad/Reshape:output:0gradients_zeros_like_cudnnrnn*
T0*j
_output_shapesX
V:                  	:         @:         @:АЦЦ
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:┘
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*4
_output_shapes"
 :                  	u
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:┼
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*'
_output_shapes
:         @y
!gradients/ExpandDims_1_grad/ShapeShape(gradients_expanddims_1_grad_shape_init_c*
T0*
_output_shapes
:╔
#gradients/ExpandDims_1_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_c_backprop:0*gradients/ExpandDims_1_grad/Shape:output:0*
T0*'
_output_shapes
:         @^
gradients/concat_1_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :Ъ
gradients/concat_1_grad/modFloorMod)gradients_concat_1_grad_mod_concat_1_axis%gradients/concat_1_grad/Rank:output:0*
T0*
_output_shapes
: h
gradients/concat_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:└j
gradients/concat_1_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:└j
gradients/concat_1_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:└j
gradients/concat_1_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:└j
gradients/concat_1_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:А j
gradients/concat_1_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:А j
gradients/concat_1_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB:А j
gradients/concat_1_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB:А i
gradients/concat_1_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB:@i
gradients/concat_1_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB:@j
 gradients/concat_1_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB:@j
 gradients/concat_1_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB:@j
 gradients/concat_1_grad/Shape_12Const*
_output_shapes
:*
dtype0*
valueB:@j
 gradients/concat_1_grad/Shape_13Const*
_output_shapes
:*
dtype0*
valueB:@j
 gradients/concat_1_grad/Shape_14Const*
_output_shapes
:*
dtype0*
valueB:@j
 gradients/concat_1_grad/Shape_15Const*
_output_shapes
:*
dtype0*
valueB:@°
$gradients/concat_1_grad/ConcatOffsetConcatOffsetgradients/concat_1_grad/mod:z:0&gradients/concat_1_grad/Shape:output:0(gradients/concat_1_grad/Shape_1:output:0(gradients/concat_1_grad/Shape_2:output:0(gradients/concat_1_grad/Shape_3:output:0(gradients/concat_1_grad/Shape_4:output:0(gradients/concat_1_grad/Shape_5:output:0(gradients/concat_1_grad/Shape_6:output:0(gradients/concat_1_grad/Shape_7:output:0(gradients/concat_1_grad/Shape_8:output:0(gradients/concat_1_grad/Shape_9:output:0)gradients/concat_1_grad/Shape_10:output:0)gradients/concat_1_grad/Shape_11:output:0)gradients/concat_1_grad/Shape_12:output:0)gradients/concat_1_grad/Shape_13:output:0)gradients/concat_1_grad/Shape_14:output:0)gradients/concat_1_grad/Shape_15:output:0*
N*t
_output_shapesb
`::::::::::::::::ь
gradients/concat_1_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:0&gradients/concat_1_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:└Ё
gradients/concat_1_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:1(gradients/concat_1_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:└Ё
gradients/concat_1_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:2(gradients/concat_1_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:└Ё
gradients/concat_1_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:3(gradients/concat_1_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:└Ё
gradients/concat_1_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:4(gradients/concat_1_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:А Ё
gradients/concat_1_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:5(gradients/concat_1_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:А Ё
gradients/concat_1_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:6(gradients/concat_1_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes	
:А Ё
gradients/concat_1_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:7(gradients/concat_1_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes	
:А я
gradients/concat_1_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:8(gradients/concat_1_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
:@я
gradients/concat_1_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:9(gradients/concat_1_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
:@Є
 gradients/concat_1_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:10)gradients/concat_1_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
:@Є
 gradients/concat_1_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:11)gradients/concat_1_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
:@Є
 gradients/concat_1_grad/Slice_12Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:12)gradients/concat_1_grad/Shape_12:output:0*
Index0*
T0*
_output_shapes
:@Є
 gradients/concat_1_grad/Slice_13Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:13)gradients/concat_1_grad/Shape_13:output:0*
Index0*
T0*
_output_shapes
:@Є
 gradients/concat_1_grad/Slice_14Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:14)gradients/concat_1_grad/Shape_14:output:0*
Index0*
T0*
_output_shapes
:@Є
 gradients/concat_1_grad/Slice_15Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:15)gradients/concat_1_grad/Shape_15:output:0*
Index0*
T0*
_output_shapes
:@m
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"@   	   б
gradients/Reshape_grad/ReshapeReshape&gradients/concat_1_grad/Slice:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes

:@	o
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"@   	   з
 gradients/Reshape_1_grad/ReshapeReshape(gradients/concat_1_grad/Slice_1:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes

:@	o
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"@   	   з
 gradients/Reshape_2_grad/ReshapeReshape(gradients/concat_1_grad/Slice_2:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes

:@	o
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"@   	   з
 gradients/Reshape_3_grad/ReshapeReshape(gradients/concat_1_grad/Slice_3:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes

:@	o
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"@   @   з
 gradients/Reshape_4_grad/ReshapeReshape(gradients/concat_1_grad/Slice_4:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:@@o
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"@   @   з
 gradients/Reshape_5_grad/ReshapeReshape(gradients/concat_1_grad/Slice_5:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:@@o
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"@   @   з
 gradients/Reshape_6_grad/ReshapeReshape(gradients/concat_1_grad/Slice_6:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:@@o
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"@   @   з
 gradients/Reshape_7_grad/ReshapeReshape(gradients/concat_1_grad/Slice_7:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes

:@@h
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:@г
 gradients/Reshape_8_grad/ReshapeReshape(gradients/concat_1_grad/Slice_8:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
:@h
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:@г
 gradients/Reshape_9_grad/ReshapeReshape(gradients/concat_1_grad/Slice_9:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
:@i
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:@ж
!gradients/Reshape_10_grad/ReshapeReshape)gradients/concat_1_grad/Slice_10:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
:@i
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:@ж
!gradients/Reshape_11_grad/ReshapeReshape)gradients/concat_1_grad/Slice_11:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
:@i
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:@ж
!gradients/Reshape_12_grad/ReshapeReshape)gradients/concat_1_grad/Slice_12:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
:@i
gradients/Reshape_13_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:@ж
!gradients/Reshape_13_grad/ReshapeReshape)gradients/concat_1_grad/Slice_13:output:0(gradients/Reshape_13_grad/Shape:output:0*
T0*
_output_shapes
:@i
gradients/Reshape_14_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:@ж
!gradients/Reshape_14_grad/ReshapeReshape)gradients/concat_1_grad/Slice_14:output:0(gradients/Reshape_14_grad/Shape:output:0*
T0*
_output_shapes
:@i
gradients/Reshape_15_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:@ж
!gradients/Reshape_15_grad/ReshapeReshape)gradients/concat_1_grad/Slice_15:output:0(gradients/Reshape_15_grad/Shape:output:0*
T0*
_output_shapes
:@Ь
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:╡
$gradients/transpose_1_grad/transpose	Transpose'gradients/Reshape_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes

:	@Ь
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:╖
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes

:	@Ь
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:╖
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes

:	@Ь
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:╖
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes

:	@Ь
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:╖
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@@Ь
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:╖
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@@Ь
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:╖
$gradients/transpose_7_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@@Ь
,gradients/transpose_8_grad/InvertPermutationInvertPermutation=gradients_transpose_8_grad_invertpermutation_transpose_8_perm*
_output_shapes
:╖
$gradients/transpose_8_grad/transpose	Transpose)gradients/Reshape_7_grad/Reshape:output:00gradients/transpose_8_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@@ч
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0*gradients/Reshape_13_grad/Reshape:output:0*gradients/Reshape_14_grad/Reshape:output:0*gradients/Reshape_15_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:Ап
gradients/split_grad/concatConcatV2(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0*
_output_shapes
:		А╡
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0(gradients/transpose_7_grad/transpose:y:0(gradients/transpose_8_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0*
_output_shapes
:	@А\
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :Т
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: f
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:Аh
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:А╩
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0*
N* 
_output_shapes
::╥
gradients/concat_grad/SliceSlice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:А╓
gradients/concat_grad/Slice_1Slice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:А{
IdentityIdentity&gradients/transpose_grad/transpose:y:0*
T0*4
_output_shapes"
 :                  	t

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0*
T0*'
_output_shapes
:         @v

Identity_2Identity,gradients/ExpandDims_1_grad/Reshape:output:0*
T0*'
_output_shapes
:         @f

Identity_3Identity$gradients/split_grad/concat:output:0*
T0*
_output_shapes
:		Аh

Identity_4Identity&gradients/split_1_grad/concat:output:0*
T0*
_output_shapes
:	@Аd

Identity_5Identity&gradients/concat_grad/Slice_1:output:0*
T0*
_output_shapes	
:А"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*(
_construction_contextkEagerRuntime*И
_input_shapesЎ
є:         @:         @:         @:         @: :         @:         @:         @:                  @::                  	:         @:         @:АЦ::         @:         @: ::::::::: : : : *=
api_implements+)lstm_8dc90f36-b787-4459-a772-dec1d233e41f*
api_preferred_deviceGPU*A
forward_function_name(&__forward_gpu_lstm_with_fallback_24954*
go_backwards( *

time_major( :- )
'
_output_shapes
:         @:1-
+
_output_shapes
:         @:-)
'
_output_shapes
:         @:-)
'
_output_shapes
:         @:

_output_shapes
: :-)
'
_output_shapes
:         @:1-
+
_output_shapes
:         @:1-
+
_output_shapes
:         @::6
4
_output_shapes"
 :                  @:	

_output_shapes
:::
6
4
_output_shapes"
 :                  	:1-
+
_output_shapes
:         @:1-
+
_output_shapes
:         @:"

_output_shapes

:АЦ: 

_output_shapes
::-)
'
_output_shapes
:         @:-)
'
_output_shapes
:         @:

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
╓k
Ь
!__inference__traced_restore_25173
file_prefix#
assignvariableop_mean:	)
assignvariableop_1_variance:	$
assignvariableop_2_count_2:	 2
assignvariableop_3_dense_kernel:	@э,
assignvariableop_4_dense_bias:	э;
(assignvariableop_5_lstm_lstm_cell_kernel:		АE
2assignvariableop_6_lstm_lstm_cell_recurrent_kernel:	@А5
&assignvariableop_7_lstm_lstm_cell_bias:	А&
assignvariableop_8_adam_iter:	 (
assignvariableop_9_adam_beta_1: )
assignvariableop_10_adam_beta_2: (
assignvariableop_11_adam_decay: 0
&assignvariableop_12_adam_learning_rate: %
assignvariableop_13_total_1: %
assignvariableop_14_count_1: #
assignvariableop_15_total: #
assignvariableop_16_count: :
'assignvariableop_17_adam_dense_kernel_m:	@э4
%assignvariableop_18_adam_dense_bias_m:	эC
0assignvariableop_19_adam_lstm_lstm_cell_kernel_m:		АM
:assignvariableop_20_adam_lstm_lstm_cell_recurrent_kernel_m:	@А=
.assignvariableop_21_adam_lstm_lstm_cell_bias_m:	А:
'assignvariableop_22_adam_dense_kernel_v:	@э4
%assignvariableop_23_adam_dense_bias_v:	эC
0assignvariableop_24_adam_lstm_lstm_cell_kernel_v:		АM
:assignvariableop_25_adam_lstm_lstm_cell_recurrent_kernel_v:	@А=
.assignvariableop_26_adam_lstm_lstm_cell_bias_v:	А
identity_28ИвAssignVariableOpвAssignVariableOp_1вAssignVariableOp_10вAssignVariableOp_11вAssignVariableOp_12вAssignVariableOp_13вAssignVariableOp_14вAssignVariableOp_15вAssignVariableOp_16вAssignVariableOp_17вAssignVariableOp_18вAssignVariableOp_19вAssignVariableOp_2вAssignVariableOp_20вAssignVariableOp_21вAssignVariableOp_22вAssignVariableOp_23вAssignVariableOp_24вAssignVariableOp_25вAssignVariableOp_26вAssignVariableOp_3вAssignVariableOp_4вAssignVariableOp_5вAssignVariableOp_6вAssignVariableOp_7вAssignVariableOp_8вAssignVariableOp_9╗
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*с
value╫B╘B4layer_with_weights-0/mean/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-0/variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-0/count/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHи
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*K
valueBB@B B B B B B B B B B B B B B B B B B B B B B B B B B B B л
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Д
_output_shapesr
p::::::::::::::::::::::::::::**
dtypes 
2		[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:А
AssignVariableOpAssignVariableOpassignvariableop_meanIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOp_1AssignVariableOpassignvariableop_1_varianceIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0	*
_output_shapes
:Й
AssignVariableOp_2AssignVariableOpassignvariableop_2_count_2Identity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:О
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_4AssignVariableOpassignvariableop_4_dense_biasIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:Ч
AssignVariableOp_5AssignVariableOp(assignvariableop_5_lstm_lstm_cell_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:б
AssignVariableOp_6AssignVariableOp2assignvariableop_6_lstm_lstm_cell_recurrent_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:Х
AssignVariableOp_7AssignVariableOp&assignvariableop_7_lstm_lstm_cell_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0	*
_output_shapes
:Л
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_iterIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:Н
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_beta_1Identity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:Р
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_beta_2Identity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:П
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_decayIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:Ч
AssignVariableOp_12AssignVariableOp&assignvariableop_12_adam_learning_rateIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_13AssignVariableOpassignvariableop_13_total_1Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_14AssignVariableOpassignvariableop_14_count_1Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOp_15AssignVariableOpassignvariableop_15_totalIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOp_16AssignVariableOpassignvariableop_16_countIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:Ш
AssignVariableOp_17AssignVariableOp'assignvariableop_17_adam_dense_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:Ц
AssignVariableOp_18AssignVariableOp%assignvariableop_18_adam_dense_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:б
AssignVariableOp_19AssignVariableOp0assignvariableop_19_adam_lstm_lstm_cell_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:л
AssignVariableOp_20AssignVariableOp:assignvariableop_20_adam_lstm_lstm_cell_recurrent_kernel_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:Я
AssignVariableOp_21AssignVariableOp.assignvariableop_21_adam_lstm_lstm_cell_bias_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:Ш
AssignVariableOp_22AssignVariableOp'assignvariableop_22_adam_dense_kernel_vIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:Ц
AssignVariableOp_23AssignVariableOp%assignvariableop_23_adam_dense_bias_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:б
AssignVariableOp_24AssignVariableOp0assignvariableop_24_adam_lstm_lstm_cell_kernel_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:л
AssignVariableOp_25AssignVariableOp:assignvariableop_25_adam_lstm_lstm_cell_recurrent_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:Я
AssignVariableOp_26AssignVariableOp.assignvariableop_26_adam_lstm_lstm_cell_bias_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 б
Identity_27Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_28IdentityIdentity_27:output:0^NoOp_1*
T0*
_output_shapes
: О
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_28Identity_28:output:0*K
_input_shapes:
8: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
Г┬
у
9__inference___backward_gpu_lstm_with_fallback_21890_22066
placeholder
placeholder_1
placeholder_2
placeholder_3
placeholder_43
/gradients_expanddims_2_grad_shape_strided_slice)
%gradients_squeeze_grad_shape_cudnnrnn+
'gradients_squeeze_1_grad_shape_cudnnrnn/
+gradients_strided_slice_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9
5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_15
1gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h,
(gradients_expanddims_1_grad_shape_init_c-
)gradients_concat_1_grad_mod_concat_1_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_permA
=gradients_transpose_7_grad_invertpermutation_transpose_7_permA
=gradients_transpose_8_grad_invertpermutation_transpose_8_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim)
%gradients_concat_grad_mod_concat_axis
identity

identity_1

identity_2

identity_3

identity_4

identity_5И^
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:         @d
gradients/grad_ys_1Identityplaceholder_1*
T0*+
_output_shapes
:         @`
gradients/grad_ys_2Identityplaceholder_2*
T0*'
_output_shapes
:         @`
gradients/grad_ys_3Identityplaceholder_3*
T0*'
_output_shapes
:         @O
gradients/grad_ys_4Identityplaceholder_4*
T0*
_output_shapes
: А
!gradients/ExpandDims_2_grad/ShapeShape/gradients_expanddims_2_grad_shape_strided_slice*
T0*
_output_shapes
:к
#gradients/ExpandDims_2_grad/ReshapeReshapegradients/grad_ys_1:output:0*gradients/ExpandDims_2_grad/Shape:output:0*
T0*'
_output_shapes
:         @q
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
:д
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*+
_output_shapes
:         @u
gradients/Squeeze_1_grad/ShapeShape'gradients_squeeze_1_grad_shape_cudnnrnn*
T0*
_output_shapes
:и
 gradients/Squeeze_1_grad/ReshapeReshapegradients/grad_ys_3:output:0'gradients/Squeeze_1_grad/Shape:output:0*
T0*+
_output_shapes
:         @┼
gradients/AddNAddNgradients/grad_ys_0:output:0,gradients/ExpandDims_2_grad/Reshape:output:0*
N*
T0*&
_class
loc:@gradients/grad_ys_0*'
_output_shapes
:         @}
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:Ж
3gradients/strided_slice_grad/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
         {
1gradients/strided_slice_grad/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 
5gradients/strided_slice_grad/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:в
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0<gradients/strided_slice_grad/StridedSliceGrad/begin:output:0:gradients/strided_slice_grad/StridedSliceGrad/end:output:0>gradients/strided_slice_grad/StridedSliceGrad/strides:output:0gradients/AddN:sum:0*
Index0*
T0*4
_output_shapes"
 :                  @*
shrink_axis_maskc
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
:м
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_11gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnn'gradients_squeeze_1_grad_shape_cudnnrnn6gradients/strided_slice_grad/StridedSliceGrad:output:0'gradients/Squeeze_grad/Reshape:output:0)gradients/Squeeze_1_grad/Reshape:output:0gradients_zeros_like_cudnnrnn*
T0*j
_output_shapesX
V:                  	:         @:         @:АЦЦ
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:┘
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*4
_output_shapes"
 :                  	u
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:┼
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*'
_output_shapes
:         @y
!gradients/ExpandDims_1_grad/ShapeShape(gradients_expanddims_1_grad_shape_init_c*
T0*
_output_shapes
:╔
#gradients/ExpandDims_1_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_c_backprop:0*gradients/ExpandDims_1_grad/Shape:output:0*
T0*'
_output_shapes
:         @^
gradients/concat_1_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :Ъ
gradients/concat_1_grad/modFloorMod)gradients_concat_1_grad_mod_concat_1_axis%gradients/concat_1_grad/Rank:output:0*
T0*
_output_shapes
: h
gradients/concat_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:└j
gradients/concat_1_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:└j
gradients/concat_1_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:└j
gradients/concat_1_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:└j
gradients/concat_1_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:А j
gradients/concat_1_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:А j
gradients/concat_1_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB:А j
gradients/concat_1_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB:А i
gradients/concat_1_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB:@i
gradients/concat_1_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB:@j
 gradients/concat_1_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB:@j
 gradients/concat_1_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB:@j
 gradients/concat_1_grad/Shape_12Const*
_output_shapes
:*
dtype0*
valueB:@j
 gradients/concat_1_grad/Shape_13Const*
_output_shapes
:*
dtype0*
valueB:@j
 gradients/concat_1_grad/Shape_14Const*
_output_shapes
:*
dtype0*
valueB:@j
 gradients/concat_1_grad/Shape_15Const*
_output_shapes
:*
dtype0*
valueB:@°
$gradients/concat_1_grad/ConcatOffsetConcatOffsetgradients/concat_1_grad/mod:z:0&gradients/concat_1_grad/Shape:output:0(gradients/concat_1_grad/Shape_1:output:0(gradients/concat_1_grad/Shape_2:output:0(gradients/concat_1_grad/Shape_3:output:0(gradients/concat_1_grad/Shape_4:output:0(gradients/concat_1_grad/Shape_5:output:0(gradients/concat_1_grad/Shape_6:output:0(gradients/concat_1_grad/Shape_7:output:0(gradients/concat_1_grad/Shape_8:output:0(gradients/concat_1_grad/Shape_9:output:0)gradients/concat_1_grad/Shape_10:output:0)gradients/concat_1_grad/Shape_11:output:0)gradients/concat_1_grad/Shape_12:output:0)gradients/concat_1_grad/Shape_13:output:0)gradients/concat_1_grad/Shape_14:output:0)gradients/concat_1_grad/Shape_15:output:0*
N*t
_output_shapesb
`::::::::::::::::ь
gradients/concat_1_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:0&gradients/concat_1_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:└Ё
gradients/concat_1_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:1(gradients/concat_1_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:└Ё
gradients/concat_1_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:2(gradients/concat_1_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:└Ё
gradients/concat_1_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:3(gradients/concat_1_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:└Ё
gradients/concat_1_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:4(gradients/concat_1_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:А Ё
gradients/concat_1_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:5(gradients/concat_1_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:А Ё
gradients/concat_1_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:6(gradients/concat_1_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes	
:А Ё
gradients/concat_1_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:7(gradients/concat_1_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes	
:А я
gradients/concat_1_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:8(gradients/concat_1_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
:@я
gradients/concat_1_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:9(gradients/concat_1_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
:@Є
 gradients/concat_1_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:10)gradients/concat_1_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
:@Є
 gradients/concat_1_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:11)gradients/concat_1_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
:@Є
 gradients/concat_1_grad/Slice_12Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:12)gradients/concat_1_grad/Shape_12:output:0*
Index0*
T0*
_output_shapes
:@Є
 gradients/concat_1_grad/Slice_13Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:13)gradients/concat_1_grad/Shape_13:output:0*
Index0*
T0*
_output_shapes
:@Є
 gradients/concat_1_grad/Slice_14Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:14)gradients/concat_1_grad/Shape_14:output:0*
Index0*
T0*
_output_shapes
:@Є
 gradients/concat_1_grad/Slice_15Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:15)gradients/concat_1_grad/Shape_15:output:0*
Index0*
T0*
_output_shapes
:@m
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"@   	   б
gradients/Reshape_grad/ReshapeReshape&gradients/concat_1_grad/Slice:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes

:@	o
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"@   	   з
 gradients/Reshape_1_grad/ReshapeReshape(gradients/concat_1_grad/Slice_1:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes

:@	o
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"@   	   з
 gradients/Reshape_2_grad/ReshapeReshape(gradients/concat_1_grad/Slice_2:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes

:@	o
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"@   	   з
 gradients/Reshape_3_grad/ReshapeReshape(gradients/concat_1_grad/Slice_3:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes

:@	o
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"@   @   з
 gradients/Reshape_4_grad/ReshapeReshape(gradients/concat_1_grad/Slice_4:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:@@o
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"@   @   з
 gradients/Reshape_5_grad/ReshapeReshape(gradients/concat_1_grad/Slice_5:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:@@o
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"@   @   з
 gradients/Reshape_6_grad/ReshapeReshape(gradients/concat_1_grad/Slice_6:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:@@o
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"@   @   з
 gradients/Reshape_7_grad/ReshapeReshape(gradients/concat_1_grad/Slice_7:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes

:@@h
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:@г
 gradients/Reshape_8_grad/ReshapeReshape(gradients/concat_1_grad/Slice_8:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
:@h
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:@г
 gradients/Reshape_9_grad/ReshapeReshape(gradients/concat_1_grad/Slice_9:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
:@i
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:@ж
!gradients/Reshape_10_grad/ReshapeReshape)gradients/concat_1_grad/Slice_10:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
:@i
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:@ж
!gradients/Reshape_11_grad/ReshapeReshape)gradients/concat_1_grad/Slice_11:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
:@i
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:@ж
!gradients/Reshape_12_grad/ReshapeReshape)gradients/concat_1_grad/Slice_12:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
:@i
gradients/Reshape_13_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:@ж
!gradients/Reshape_13_grad/ReshapeReshape)gradients/concat_1_grad/Slice_13:output:0(gradients/Reshape_13_grad/Shape:output:0*
T0*
_output_shapes
:@i
gradients/Reshape_14_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:@ж
!gradients/Reshape_14_grad/ReshapeReshape)gradients/concat_1_grad/Slice_14:output:0(gradients/Reshape_14_grad/Shape:output:0*
T0*
_output_shapes
:@i
gradients/Reshape_15_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:@ж
!gradients/Reshape_15_grad/ReshapeReshape)gradients/concat_1_grad/Slice_15:output:0(gradients/Reshape_15_grad/Shape:output:0*
T0*
_output_shapes
:@Ь
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:╡
$gradients/transpose_1_grad/transpose	Transpose'gradients/Reshape_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes

:	@Ь
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:╖
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes

:	@Ь
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:╖
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes

:	@Ь
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:╖
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes

:	@Ь
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:╖
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@@Ь
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:╖
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@@Ь
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:╖
$gradients/transpose_7_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@@Ь
,gradients/transpose_8_grad/InvertPermutationInvertPermutation=gradients_transpose_8_grad_invertpermutation_transpose_8_perm*
_output_shapes
:╖
$gradients/transpose_8_grad/transpose	Transpose)gradients/Reshape_7_grad/Reshape:output:00gradients/transpose_8_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@@ч
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0*gradients/Reshape_13_grad/Reshape:output:0*gradients/Reshape_14_grad/Reshape:output:0*gradients/Reshape_15_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:Ап
gradients/split_grad/concatConcatV2(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0*
_output_shapes
:		А╡
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0(gradients/transpose_7_grad/transpose:y:0(gradients/transpose_8_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0*
_output_shapes
:	@А\
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :Т
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: f
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:Аh
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:А╩
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0*
N* 
_output_shapes
::╥
gradients/concat_grad/SliceSlice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:А╓
gradients/concat_grad/Slice_1Slice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:А{
IdentityIdentity&gradients/transpose_grad/transpose:y:0*
T0*4
_output_shapes"
 :                  	t

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0*
T0*'
_output_shapes
:         @v

Identity_2Identity,gradients/ExpandDims_1_grad/Reshape:output:0*
T0*'
_output_shapes
:         @f

Identity_3Identity$gradients/split_grad/concat:output:0*
T0*
_output_shapes
:		Аh

Identity_4Identity&gradients/split_1_grad/concat:output:0*
T0*
_output_shapes
:	@Аd

Identity_5Identity&gradients/concat_grad/Slice_1:output:0*
T0*
_output_shapes	
:А"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*(
_construction_contextkEagerRuntime*И
_input_shapesЎ
є:         @:         @:         @:         @: :         @:         @:         @:                  @::                  	:         @:         @:АЦ::         @:         @: ::::::::: : : : *=
api_implements+)lstm_d4300e95-cf4b-4bbb-ba38-7f8ecfb78490*
api_preferred_deviceGPU*A
forward_function_name(&__forward_gpu_lstm_with_fallback_22065*
go_backwards( *

time_major( :- )
'
_output_shapes
:         @:1-
+
_output_shapes
:         @:-)
'
_output_shapes
:         @:-)
'
_output_shapes
:         @:

_output_shapes
: :-)
'
_output_shapes
:         @:1-
+
_output_shapes
:         @:1-
+
_output_shapes
:         @::6
4
_output_shapes"
 :                  @:	

_output_shapes
:::
6
4
_output_shapes"
 :                  	:1-
+
_output_shapes
:         @:1-
+
_output_shapes
:         @:"

_output_shapes

:АЦ: 

_output_shapes
::-)
'
_output_shapes
:         @:-)
'
_output_shapes
:         @:

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
─

Ъ
*__inference_sequential_layer_call_fn_21616
normalization_input
unknown
	unknown_0
	unknown_1:		А
	unknown_2:	@А
	unknown_3:	А
	unknown_4:	@э
	unknown_5:	э
identityИвStatefulPartitionedCallз
StatefulPartitionedCallStatefulPartitionedCallnormalization_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         э*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_21599p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         э`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:'                           :	:	: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:r n
=
_output_shapes+
):'                           
-
_user_specified_namenormalization_input:($
"
_output_shapes
:	:($
"
_output_shapes
:	
Э

Н
*__inference_sequential_layer_call_fn_22313

inputs
unknown
	unknown_0
	unknown_1:		А
	unknown_2:	@А
	unknown_3:	А
	unknown_4:	@э
	unknown_5:	э
identityИвStatefulPartitionedCallЪ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         э*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_22121p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         э`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:'                           :	:	: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:e a
=
_output_shapes+
):'                           
 
_user_specified_nameinputs:($
"
_output_shapes
:	:($
"
_output_shapes
:	
М	
╝
while_cond_19884
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice3
/while_while_cond_19884___redundant_placeholder03
/while_while_cond_19884___redundant_placeholder13
/while_while_cond_19884___redundant_placeholder23
/while_while_cond_19884___redundant_placeholder3
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :         @:         @: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         @:-)
'
_output_shapes
:         @:

_output_shapes
: :

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
Г┬
у
9__inference___backward_gpu_lstm_with_fallback_23492_23668
placeholder
placeholder_1
placeholder_2
placeholder_3
placeholder_43
/gradients_expanddims_2_grad_shape_strided_slice)
%gradients_squeeze_grad_shape_cudnnrnn+
'gradients_squeeze_1_grad_shape_cudnnrnn/
+gradients_strided_slice_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9
5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_15
1gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h,
(gradients_expanddims_1_grad_shape_init_c-
)gradients_concat_1_grad_mod_concat_1_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_permA
=gradients_transpose_7_grad_invertpermutation_transpose_7_permA
=gradients_transpose_8_grad_invertpermutation_transpose_8_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim)
%gradients_concat_grad_mod_concat_axis
identity

identity_1

identity_2

identity_3

identity_4

identity_5И^
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:         @d
gradients/grad_ys_1Identityplaceholder_1*
T0*+
_output_shapes
:         @`
gradients/grad_ys_2Identityplaceholder_2*
T0*'
_output_shapes
:         @`
gradients/grad_ys_3Identityplaceholder_3*
T0*'
_output_shapes
:         @O
gradients/grad_ys_4Identityplaceholder_4*
T0*
_output_shapes
: А
!gradients/ExpandDims_2_grad/ShapeShape/gradients_expanddims_2_grad_shape_strided_slice*
T0*
_output_shapes
:к
#gradients/ExpandDims_2_grad/ReshapeReshapegradients/grad_ys_1:output:0*gradients/ExpandDims_2_grad/Shape:output:0*
T0*'
_output_shapes
:         @q
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
:д
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*+
_output_shapes
:         @u
gradients/Squeeze_1_grad/ShapeShape'gradients_squeeze_1_grad_shape_cudnnrnn*
T0*
_output_shapes
:и
 gradients/Squeeze_1_grad/ReshapeReshapegradients/grad_ys_3:output:0'gradients/Squeeze_1_grad/Shape:output:0*
T0*+
_output_shapes
:         @┼
gradients/AddNAddNgradients/grad_ys_0:output:0,gradients/ExpandDims_2_grad/Reshape:output:0*
N*
T0*&
_class
loc:@gradients/grad_ys_0*'
_output_shapes
:         @}
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:Ж
3gradients/strided_slice_grad/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
         {
1gradients/strided_slice_grad/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 
5gradients/strided_slice_grad/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:в
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0<gradients/strided_slice_grad/StridedSliceGrad/begin:output:0:gradients/strided_slice_grad/StridedSliceGrad/end:output:0>gradients/strided_slice_grad/StridedSliceGrad/strides:output:0gradients/AddN:sum:0*
Index0*
T0*4
_output_shapes"
 :                  @*
shrink_axis_maskc
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
:м
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_11gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnn'gradients_squeeze_1_grad_shape_cudnnrnn6gradients/strided_slice_grad/StridedSliceGrad:output:0'gradients/Squeeze_grad/Reshape:output:0)gradients/Squeeze_1_grad/Reshape:output:0gradients_zeros_like_cudnnrnn*
T0*j
_output_shapesX
V:                  	:         @:         @:АЦЦ
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:┘
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*4
_output_shapes"
 :                  	u
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:┼
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*'
_output_shapes
:         @y
!gradients/ExpandDims_1_grad/ShapeShape(gradients_expanddims_1_grad_shape_init_c*
T0*
_output_shapes
:╔
#gradients/ExpandDims_1_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_c_backprop:0*gradients/ExpandDims_1_grad/Shape:output:0*
T0*'
_output_shapes
:         @^
gradients/concat_1_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :Ъ
gradients/concat_1_grad/modFloorMod)gradients_concat_1_grad_mod_concat_1_axis%gradients/concat_1_grad/Rank:output:0*
T0*
_output_shapes
: h
gradients/concat_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:└j
gradients/concat_1_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:└j
gradients/concat_1_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:└j
gradients/concat_1_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:└j
gradients/concat_1_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:А j
gradients/concat_1_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:А j
gradients/concat_1_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB:А j
gradients/concat_1_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB:А i
gradients/concat_1_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB:@i
gradients/concat_1_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB:@j
 gradients/concat_1_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB:@j
 gradients/concat_1_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB:@j
 gradients/concat_1_grad/Shape_12Const*
_output_shapes
:*
dtype0*
valueB:@j
 gradients/concat_1_grad/Shape_13Const*
_output_shapes
:*
dtype0*
valueB:@j
 gradients/concat_1_grad/Shape_14Const*
_output_shapes
:*
dtype0*
valueB:@j
 gradients/concat_1_grad/Shape_15Const*
_output_shapes
:*
dtype0*
valueB:@°
$gradients/concat_1_grad/ConcatOffsetConcatOffsetgradients/concat_1_grad/mod:z:0&gradients/concat_1_grad/Shape:output:0(gradients/concat_1_grad/Shape_1:output:0(gradients/concat_1_grad/Shape_2:output:0(gradients/concat_1_grad/Shape_3:output:0(gradients/concat_1_grad/Shape_4:output:0(gradients/concat_1_grad/Shape_5:output:0(gradients/concat_1_grad/Shape_6:output:0(gradients/concat_1_grad/Shape_7:output:0(gradients/concat_1_grad/Shape_8:output:0(gradients/concat_1_grad/Shape_9:output:0)gradients/concat_1_grad/Shape_10:output:0)gradients/concat_1_grad/Shape_11:output:0)gradients/concat_1_grad/Shape_12:output:0)gradients/concat_1_grad/Shape_13:output:0)gradients/concat_1_grad/Shape_14:output:0)gradients/concat_1_grad/Shape_15:output:0*
N*t
_output_shapesb
`::::::::::::::::ь
gradients/concat_1_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:0&gradients/concat_1_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:└Ё
gradients/concat_1_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:1(gradients/concat_1_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:└Ё
gradients/concat_1_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:2(gradients/concat_1_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:└Ё
gradients/concat_1_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:3(gradients/concat_1_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:└Ё
gradients/concat_1_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:4(gradients/concat_1_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:А Ё
gradients/concat_1_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:5(gradients/concat_1_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:А Ё
gradients/concat_1_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:6(gradients/concat_1_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes	
:А Ё
gradients/concat_1_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:7(gradients/concat_1_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes	
:А я
gradients/concat_1_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:8(gradients/concat_1_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
:@я
gradients/concat_1_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:9(gradients/concat_1_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
:@Є
 gradients/concat_1_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:10)gradients/concat_1_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
:@Є
 gradients/concat_1_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:11)gradients/concat_1_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
:@Є
 gradients/concat_1_grad/Slice_12Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:12)gradients/concat_1_grad/Shape_12:output:0*
Index0*
T0*
_output_shapes
:@Є
 gradients/concat_1_grad/Slice_13Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:13)gradients/concat_1_grad/Shape_13:output:0*
Index0*
T0*
_output_shapes
:@Є
 gradients/concat_1_grad/Slice_14Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:14)gradients/concat_1_grad/Shape_14:output:0*
Index0*
T0*
_output_shapes
:@Є
 gradients/concat_1_grad/Slice_15Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:15)gradients/concat_1_grad/Shape_15:output:0*
Index0*
T0*
_output_shapes
:@m
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"@   	   б
gradients/Reshape_grad/ReshapeReshape&gradients/concat_1_grad/Slice:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes

:@	o
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"@   	   з
 gradients/Reshape_1_grad/ReshapeReshape(gradients/concat_1_grad/Slice_1:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes

:@	o
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"@   	   з
 gradients/Reshape_2_grad/ReshapeReshape(gradients/concat_1_grad/Slice_2:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes

:@	o
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"@   	   з
 gradients/Reshape_3_grad/ReshapeReshape(gradients/concat_1_grad/Slice_3:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes

:@	o
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"@   @   з
 gradients/Reshape_4_grad/ReshapeReshape(gradients/concat_1_grad/Slice_4:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:@@o
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"@   @   з
 gradients/Reshape_5_grad/ReshapeReshape(gradients/concat_1_grad/Slice_5:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:@@o
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"@   @   з
 gradients/Reshape_6_grad/ReshapeReshape(gradients/concat_1_grad/Slice_6:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:@@o
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"@   @   з
 gradients/Reshape_7_grad/ReshapeReshape(gradients/concat_1_grad/Slice_7:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes

:@@h
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:@г
 gradients/Reshape_8_grad/ReshapeReshape(gradients/concat_1_grad/Slice_8:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
:@h
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:@г
 gradients/Reshape_9_grad/ReshapeReshape(gradients/concat_1_grad/Slice_9:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
:@i
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:@ж
!gradients/Reshape_10_grad/ReshapeReshape)gradients/concat_1_grad/Slice_10:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
:@i
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:@ж
!gradients/Reshape_11_grad/ReshapeReshape)gradients/concat_1_grad/Slice_11:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
:@i
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:@ж
!gradients/Reshape_12_grad/ReshapeReshape)gradients/concat_1_grad/Slice_12:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
:@i
gradients/Reshape_13_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:@ж
!gradients/Reshape_13_grad/ReshapeReshape)gradients/concat_1_grad/Slice_13:output:0(gradients/Reshape_13_grad/Shape:output:0*
T0*
_output_shapes
:@i
gradients/Reshape_14_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:@ж
!gradients/Reshape_14_grad/ReshapeReshape)gradients/concat_1_grad/Slice_14:output:0(gradients/Reshape_14_grad/Shape:output:0*
T0*
_output_shapes
:@i
gradients/Reshape_15_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:@ж
!gradients/Reshape_15_grad/ReshapeReshape)gradients/concat_1_grad/Slice_15:output:0(gradients/Reshape_15_grad/Shape:output:0*
T0*
_output_shapes
:@Ь
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:╡
$gradients/transpose_1_grad/transpose	Transpose'gradients/Reshape_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes

:	@Ь
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:╖
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes

:	@Ь
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:╖
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes

:	@Ь
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:╖
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes

:	@Ь
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:╖
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@@Ь
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:╖
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@@Ь
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:╖
$gradients/transpose_7_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@@Ь
,gradients/transpose_8_grad/InvertPermutationInvertPermutation=gradients_transpose_8_grad_invertpermutation_transpose_8_perm*
_output_shapes
:╖
$gradients/transpose_8_grad/transpose	Transpose)gradients/Reshape_7_grad/Reshape:output:00gradients/transpose_8_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@@ч
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0*gradients/Reshape_13_grad/Reshape:output:0*gradients/Reshape_14_grad/Reshape:output:0*gradients/Reshape_15_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:Ап
gradients/split_grad/concatConcatV2(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0*
_output_shapes
:		А╡
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0(gradients/transpose_7_grad/transpose:y:0(gradients/transpose_8_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0*
_output_shapes
:	@А\
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :Т
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: f
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:Аh
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:А╩
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0*
N* 
_output_shapes
::╥
gradients/concat_grad/SliceSlice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:А╓
gradients/concat_grad/Slice_1Slice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:А{
IdentityIdentity&gradients/transpose_grad/transpose:y:0*
T0*4
_output_shapes"
 :                  	t

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0*
T0*'
_output_shapes
:         @v

Identity_2Identity,gradients/ExpandDims_1_grad/Reshape:output:0*
T0*'
_output_shapes
:         @f

Identity_3Identity$gradients/split_grad/concat:output:0*
T0*
_output_shapes
:		Аh

Identity_4Identity&gradients/split_1_grad/concat:output:0*
T0*
_output_shapes
:	@Аd

Identity_5Identity&gradients/concat_grad/Slice_1:output:0*
T0*
_output_shapes	
:А"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*(
_construction_contextkEagerRuntime*И
_input_shapesЎ
є:         @:         @:         @:         @: :         @:         @:         @:                  @::                  	:         @:         @:АЦ::         @:         @: ::::::::: : : : *=
api_implements+)lstm_d60d0f35-f731-4def-8ea6-8cb0eeff7cf2*
api_preferred_deviceGPU*A
forward_function_name(&__forward_gpu_lstm_with_fallback_23667*
go_backwards( *

time_major( :- )
'
_output_shapes
:         @:1-
+
_output_shapes
:         @:-)
'
_output_shapes
:         @:-)
'
_output_shapes
:         @:

_output_shapes
: :-)
'
_output_shapes
:         @:1-
+
_output_shapes
:         @:1-
+
_output_shapes
:         @::6
4
_output_shapes"
 :                  @:	

_output_shapes
:::
6
4
_output_shapes"
 :                  	:1-
+
_output_shapes
:         @:1-
+
_output_shapes
:         @:"

_output_shapes

:АЦ: 

_output_shapes
::-)
'
_output_shapes
:         @:-)
'
_output_shapes
:         @:

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Э

Н
*__inference_sequential_layer_call_fn_22294

inputs
unknown
	unknown_0
	unknown_1:		А
	unknown_2:	@А
	unknown_3:	А
	unknown_4:	@э
	unknown_5:	э
identityИвStatefulPartitionedCallЪ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         э*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_21599p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         э`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:'                           :	:	: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:e a
=
_output_shapes+
):'                           
 
_user_specified_nameinputs:($
"
_output_shapes
:	:($
"
_output_shapes
:	
─

Ъ
*__inference_sequential_layer_call_fn_22157
normalization_input
unknown
	unknown_0
	unknown_1:		А
	unknown_2:	@А
	unknown_3:	А
	unknown_4:	@э
	unknown_5:	э
identityИвStatefulPartitionedCallз
StatefulPartitionedCallStatefulPartitionedCallnormalization_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         э*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_22121p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         э`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:'                           :	:	: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:r n
=
_output_shapes+
):'                           
-
_user_specified_namenormalization_input:($
"
_output_shapes
:	:($
"
_output_shapes
:	
к
╗
?__inference_lstm_layer_call_and_return_conditional_losses_23670
inputs_0/
read_readvariableop_resource:		А1
read_1_readvariableop_resource:	@А-
read_2_readvariableop_resource:	А

identity_3ИвRead/ReadVariableOpвRead_1/ReadVariableOpвRead_2/ReadVariableOp=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:         @R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:         @q
Read/ReadVariableOpReadVariableOpread_readvariableop_resource*
_output_shapes
:		А*
dtype0[
IdentityIdentityRead/ReadVariableOp:value:0*
T0*
_output_shapes
:		Аu
Read_1/ReadVariableOpReadVariableOpread_1_readvariableop_resource*
_output_shapes
:	@А*
dtype0_

Identity_1IdentityRead_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	@Аq
Read_2/ReadVariableOpReadVariableOpread_2_readvariableop_resource*
_output_shapes	
:А*
dtype0[

Identity_2IdentityRead_2/ReadVariableOp:value:0*
T0*
_output_shapes	
:А│
PartitionedCallPartitionedCallinputs_0zeros:output:0zeros_1:output:0Identity:output:0Identity_1:output:0Identity_2:output:0*
Tin

2*
Tout	
2*
_collective_manager_ids
 *f
_output_shapesT
R:         @:         @:         @:         @: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *(
f#R!
__inference_standard_lstm_23397i

Identity_3IdentityPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         @М
NoOpNoOp^Read/ReadVariableOp^Read_1/ReadVariableOp^Read_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  	: : : 2*
Read/ReadVariableOpRead/ReadVariableOp2.
Read_1/ReadVariableOpRead_1/ReadVariableOp2.
Read_2/ReadVariableOpRead_2/ReadVariableOp:^ Z
4
_output_shapes"
 :                  	
"
_user_specified_name
inputs/0
╪)
╬
while_body_20324
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_bias_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel#
while_matmul_1_recurrent_kernel
while_biasadd_biasИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    	   ж
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         	*
element_dtype0Т
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*(
_output_shapes
:         АГ
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*(
_output_shapes
:         Аw
	while/addAddV2while/MatMul:product:0while/MatMul_1:product:0*
T0*(
_output_shapes
:         Аp
while/BiasAddBiasAddwhile/add:z:0while_biasadd_bias_0*
T0*(
_output_shapes
:         АW
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :╚
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*`
_output_shapesN
L:         @:         @:         @:         @*
	num_split`
while/SigmoidSigmoidwhile/split:output:0*
T0*'
_output_shapes
:         @b
while/Sigmoid_1Sigmoidwhile/split:output:1*
T0*'
_output_shapes
:         @l
	while/mulMulwhile/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:         @Z

while/TanhTanhwhile/split:output:2*
T0*'
_output_shapes
:         @g
while/mul_1Mulwhile/Sigmoid:y:0while/Tanh:y:0*
T0*'
_output_shapes
:         @f
while/add_1AddV2while/mul:z:0while/mul_1:z:0*
T0*'
_output_shapes
:         @b
while/Sigmoid_2Sigmoidwhile/split:output:3*
T0*'
_output_shapes
:         @W
while/Tanh_1Tanhwhile/add_1:z:0*
T0*'
_output_shapes
:         @k
while/mul_2Mulwhile/Sigmoid_2:y:0while/Tanh_1:y:0*
T0*'
_output_shapes
:         @r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : р
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/mul_2:z:0*
_output_shapes
: *
element_dtype0:щш╥O
while/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :`
while/add_2AddV2while_placeholderwhile/add_2/y:output:0*
T0*
_output_shapes
: O
while/add_3/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_3AddV2while_while_loop_counterwhile/add_3/y:output:0*
T0*
_output_shapes
: L
while/IdentityIdentitywhile/add_3:z:0*
T0*
_output_shapes
: ]
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: N
while/Identity_2Identitywhile/add_2:z:0*
T0*
_output_shapes
: y
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: _
while/Identity_4Identitywhile/mul_2:z:0*
T0*'
_output_shapes
:         @_
while/Identity_5Identitywhile/add_1:z:0*
T0*'
_output_shapes
:         @"*
while_biasadd_biaswhile_biasadd_bias_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O: : : : :         @:         @: : :		А:	@А:А: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         @:-)
'
_output_shapes
:         @:

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:		А:%	!

_output_shapes
:	@А:!


_output_shapes	
:А
к
ы
E__inference_sequential_layer_call_and_return_conditional_losses_22121

inputs
normalization_sub_y
normalization_sqrt_x

lstm_22108:		А

lstm_22110:	@А

lstm_22112:	А
dense_22115:	@э
dense_22117:	э
identityИвdense/StatefulPartitionedCallвlstm/StatefulPartitionedCallt
normalization/subSubinputsnormalization_sub_y*
T0*4
_output_shapes"
 :                  	]
normalization/SqrtSqrtnormalization_sqrt_x*
T0*"
_output_shapes
:	\
normalization/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *Х┐╓3З
normalization/MaximumMaximumnormalization/Sqrt:y:0 normalization/Maximum/y:output:0*
T0*"
_output_shapes
:	С
normalization/truedivRealDivnormalization/sub:z:0normalization/Maximum:z:0*
T0*4
_output_shapes"
 :                  	■
lstm/StatefulPartitionedCallStatefulPartitionedCallnormalization/truediv:z:0
lstm_22108
lstm_22110
lstm_22112*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *H
fCRA
?__inference_lstm_layer_call_and_return_conditional_losses_22068Б
dense/StatefulPartitionedCallStatefulPartitionedCall%lstm/StatefulPartitionedCall:output:0dense_22115dense_22117*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         э*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_21592v
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         эЕ
NoOpNoOp^dense/StatefulPartitionedCall^lstm/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:'                           :	:	: : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2<
lstm/StatefulPartitionedCalllstm/StatefulPartitionedCall:e a
=
_output_shapes+
):'                           
 
_user_specified_nameinputs:($
"
_output_shapes
:	:($
"
_output_shapes
:	
╟@
╦
(__inference_gpu_lstm_with_fallback_24349

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4Иc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :                  	P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : o

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*+
_output_shapes
:         @R
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : s
ExpandDims_1
ExpandDimsinit_cExpandDims_1/dim:output:0*
T0*+
_output_shapes
:         @Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :И
splitSplitsplit/split_dim:output:0kernel*
T0*<
_output_shapes*
(:	@:	@:	@:	@*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ц
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*<
_output_shapes*
(:@@:@@:@@:@@*
	num_splitY

zeros_likeConst*
_output_shapes	
:А*
dtype0*
valueBА*    M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : r
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes	
:АS
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Э
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*D
_output_shapes2
0:@:@:@:@:@:@:@:@*
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
         a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes

:@	Y
ReshapeReshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:└a
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes

:@	[
	Reshape_1Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:└a
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

:@	[
	Reshape_2Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:└a
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes

:@	[
	Reshape_3Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:└a
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:@@[
	Reshape_4Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:А a
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0*
_output_shapes

:@@[
	Reshape_5Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:А a
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0*
_output_shapes

:@@[
	Reshape_6Reshapetranspose_7:y:0Const:output:0*
T0*
_output_shapes	
:А a
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0*
_output_shapes

:@@[
	Reshape_7Reshapetranspose_8:y:0Const:output:0*
T0*
_output_shapes	
:А [
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
:@[
	Reshape_9Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
:@\

Reshape_10Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
:@\

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
:@\

Reshape_12Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
:@\

Reshape_13Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
:@\

Reshape_14Reshapesplit_2:output:6Const:output:0*
T0*
_output_shapes
:@\

Reshape_15Reshapesplit_2:output:7Const:output:0*
T0*
_output_shapes
:@O
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : а
concat_1ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes

:АЦ╙
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1:output:0*
T0*f
_output_shapesT
R:                  @:         @:         @:f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
         _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:х
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         @*
shrink_axis_maskp
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:         @*
squeeze_dims
 r
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*'
_output_shapes
:         @*
squeeze_dims
 R
ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B :Г
ExpandDims_2
ExpandDimsstrided_slice:output:0ExpandDims_2/dim:output:0*
T0*+
_output_shapes
:         @[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @^
IdentityIdentitystrided_slice:output:0*
T0*'
_output_shapes
:         @c

Identity_1IdentityExpandDims_2:output:0*
T0*+
_output_shapes
:         @Z

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:         @\

Identity_3IdentitySqueeze_1:output:0*
T0*'
_output_shapes
:         @I

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:                  	:         @:         @:		А:	@А:А*=
api_implements+)lstm_e5590ddf-237b-46f1-9660-fe95f9321bd6*
api_preferred_deviceGPU*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :                  	
 
_user_specified_nameinputs:OK
'
_output_shapes
:         @
 
_user_specified_nameinit_h:OK
'
_output_shapes
:         @
 
_user_specified_nameinit_c:GC

_output_shapes
:		А
 
_user_specified_namekernel:QM

_output_shapes
:	@А
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:А

_user_specified_namebias
Г┬
у
9__inference___backward_gpu_lstm_with_fallback_22571_22747
placeholder
placeholder_1
placeholder_2
placeholder_3
placeholder_43
/gradients_expanddims_2_grad_shape_strided_slice)
%gradients_squeeze_grad_shape_cudnnrnn+
'gradients_squeeze_1_grad_shape_cudnnrnn/
+gradients_strided_slice_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9
5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_15
1gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h,
(gradients_expanddims_1_grad_shape_init_c-
)gradients_concat_1_grad_mod_concat_1_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_permA
=gradients_transpose_7_grad_invertpermutation_transpose_7_permA
=gradients_transpose_8_grad_invertpermutation_transpose_8_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim)
%gradients_concat_grad_mod_concat_axis
identity

identity_1

identity_2

identity_3

identity_4

identity_5И^
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:         @d
gradients/grad_ys_1Identityplaceholder_1*
T0*+
_output_shapes
:         @`
gradients/grad_ys_2Identityplaceholder_2*
T0*'
_output_shapes
:         @`
gradients/grad_ys_3Identityplaceholder_3*
T0*'
_output_shapes
:         @O
gradients/grad_ys_4Identityplaceholder_4*
T0*
_output_shapes
: А
!gradients/ExpandDims_2_grad/ShapeShape/gradients_expanddims_2_grad_shape_strided_slice*
T0*
_output_shapes
:к
#gradients/ExpandDims_2_grad/ReshapeReshapegradients/grad_ys_1:output:0*gradients/ExpandDims_2_grad/Shape:output:0*
T0*'
_output_shapes
:         @q
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
:д
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*+
_output_shapes
:         @u
gradients/Squeeze_1_grad/ShapeShape'gradients_squeeze_1_grad_shape_cudnnrnn*
T0*
_output_shapes
:и
 gradients/Squeeze_1_grad/ReshapeReshapegradients/grad_ys_3:output:0'gradients/Squeeze_1_grad/Shape:output:0*
T0*+
_output_shapes
:         @┼
gradients/AddNAddNgradients/grad_ys_0:output:0,gradients/ExpandDims_2_grad/Reshape:output:0*
N*
T0*&
_class
loc:@gradients/grad_ys_0*'
_output_shapes
:         @}
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:Ж
3gradients/strided_slice_grad/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
         {
1gradients/strided_slice_grad/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 
5gradients/strided_slice_grad/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:в
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0<gradients/strided_slice_grad/StridedSliceGrad/begin:output:0:gradients/strided_slice_grad/StridedSliceGrad/end:output:0>gradients/strided_slice_grad/StridedSliceGrad/strides:output:0gradients/AddN:sum:0*
Index0*
T0*4
_output_shapes"
 :                  @*
shrink_axis_maskc
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
:м
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_11gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnn'gradients_squeeze_1_grad_shape_cudnnrnn6gradients/strided_slice_grad/StridedSliceGrad:output:0'gradients/Squeeze_grad/Reshape:output:0)gradients/Squeeze_1_grad/Reshape:output:0gradients_zeros_like_cudnnrnn*
T0*j
_output_shapesX
V:                  	:         @:         @:АЦЦ
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:┘
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*4
_output_shapes"
 :                  	u
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:┼
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*'
_output_shapes
:         @y
!gradients/ExpandDims_1_grad/ShapeShape(gradients_expanddims_1_grad_shape_init_c*
T0*
_output_shapes
:╔
#gradients/ExpandDims_1_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_c_backprop:0*gradients/ExpandDims_1_grad/Shape:output:0*
T0*'
_output_shapes
:         @^
gradients/concat_1_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :Ъ
gradients/concat_1_grad/modFloorMod)gradients_concat_1_grad_mod_concat_1_axis%gradients/concat_1_grad/Rank:output:0*
T0*
_output_shapes
: h
gradients/concat_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:└j
gradients/concat_1_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:└j
gradients/concat_1_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:└j
gradients/concat_1_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:└j
gradients/concat_1_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:А j
gradients/concat_1_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:А j
gradients/concat_1_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB:А j
gradients/concat_1_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB:А i
gradients/concat_1_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB:@i
gradients/concat_1_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB:@j
 gradients/concat_1_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB:@j
 gradients/concat_1_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB:@j
 gradients/concat_1_grad/Shape_12Const*
_output_shapes
:*
dtype0*
valueB:@j
 gradients/concat_1_grad/Shape_13Const*
_output_shapes
:*
dtype0*
valueB:@j
 gradients/concat_1_grad/Shape_14Const*
_output_shapes
:*
dtype0*
valueB:@j
 gradients/concat_1_grad/Shape_15Const*
_output_shapes
:*
dtype0*
valueB:@°
$gradients/concat_1_grad/ConcatOffsetConcatOffsetgradients/concat_1_grad/mod:z:0&gradients/concat_1_grad/Shape:output:0(gradients/concat_1_grad/Shape_1:output:0(gradients/concat_1_grad/Shape_2:output:0(gradients/concat_1_grad/Shape_3:output:0(gradients/concat_1_grad/Shape_4:output:0(gradients/concat_1_grad/Shape_5:output:0(gradients/concat_1_grad/Shape_6:output:0(gradients/concat_1_grad/Shape_7:output:0(gradients/concat_1_grad/Shape_8:output:0(gradients/concat_1_grad/Shape_9:output:0)gradients/concat_1_grad/Shape_10:output:0)gradients/concat_1_grad/Shape_11:output:0)gradients/concat_1_grad/Shape_12:output:0)gradients/concat_1_grad/Shape_13:output:0)gradients/concat_1_grad/Shape_14:output:0)gradients/concat_1_grad/Shape_15:output:0*
N*t
_output_shapesb
`::::::::::::::::ь
gradients/concat_1_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:0&gradients/concat_1_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:└Ё
gradients/concat_1_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:1(gradients/concat_1_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:└Ё
gradients/concat_1_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:2(gradients/concat_1_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:└Ё
gradients/concat_1_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:3(gradients/concat_1_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:└Ё
gradients/concat_1_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:4(gradients/concat_1_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:А Ё
gradients/concat_1_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:5(gradients/concat_1_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:А Ё
gradients/concat_1_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:6(gradients/concat_1_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes	
:А Ё
gradients/concat_1_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:7(gradients/concat_1_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes	
:А я
gradients/concat_1_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:8(gradients/concat_1_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
:@я
gradients/concat_1_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:9(gradients/concat_1_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
:@Є
 gradients/concat_1_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:10)gradients/concat_1_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
:@Є
 gradients/concat_1_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:11)gradients/concat_1_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
:@Є
 gradients/concat_1_grad/Slice_12Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:12)gradients/concat_1_grad/Shape_12:output:0*
Index0*
T0*
_output_shapes
:@Є
 gradients/concat_1_grad/Slice_13Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:13)gradients/concat_1_grad/Shape_13:output:0*
Index0*
T0*
_output_shapes
:@Є
 gradients/concat_1_grad/Slice_14Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:14)gradients/concat_1_grad/Shape_14:output:0*
Index0*
T0*
_output_shapes
:@Є
 gradients/concat_1_grad/Slice_15Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:15)gradients/concat_1_grad/Shape_15:output:0*
Index0*
T0*
_output_shapes
:@m
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"@   	   б
gradients/Reshape_grad/ReshapeReshape&gradients/concat_1_grad/Slice:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes

:@	o
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"@   	   з
 gradients/Reshape_1_grad/ReshapeReshape(gradients/concat_1_grad/Slice_1:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes

:@	o
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"@   	   з
 gradients/Reshape_2_grad/ReshapeReshape(gradients/concat_1_grad/Slice_2:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes

:@	o
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"@   	   з
 gradients/Reshape_3_grad/ReshapeReshape(gradients/concat_1_grad/Slice_3:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes

:@	o
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"@   @   з
 gradients/Reshape_4_grad/ReshapeReshape(gradients/concat_1_grad/Slice_4:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:@@o
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"@   @   з
 gradients/Reshape_5_grad/ReshapeReshape(gradients/concat_1_grad/Slice_5:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:@@o
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"@   @   з
 gradients/Reshape_6_grad/ReshapeReshape(gradients/concat_1_grad/Slice_6:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:@@o
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"@   @   з
 gradients/Reshape_7_grad/ReshapeReshape(gradients/concat_1_grad/Slice_7:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes

:@@h
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:@г
 gradients/Reshape_8_grad/ReshapeReshape(gradients/concat_1_grad/Slice_8:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
:@h
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:@г
 gradients/Reshape_9_grad/ReshapeReshape(gradients/concat_1_grad/Slice_9:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
:@i
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:@ж
!gradients/Reshape_10_grad/ReshapeReshape)gradients/concat_1_grad/Slice_10:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
:@i
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:@ж
!gradients/Reshape_11_grad/ReshapeReshape)gradients/concat_1_grad/Slice_11:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
:@i
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:@ж
!gradients/Reshape_12_grad/ReshapeReshape)gradients/concat_1_grad/Slice_12:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
:@i
gradients/Reshape_13_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:@ж
!gradients/Reshape_13_grad/ReshapeReshape)gradients/concat_1_grad/Slice_13:output:0(gradients/Reshape_13_grad/Shape:output:0*
T0*
_output_shapes
:@i
gradients/Reshape_14_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:@ж
!gradients/Reshape_14_grad/ReshapeReshape)gradients/concat_1_grad/Slice_14:output:0(gradients/Reshape_14_grad/Shape:output:0*
T0*
_output_shapes
:@i
gradients/Reshape_15_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:@ж
!gradients/Reshape_15_grad/ReshapeReshape)gradients/concat_1_grad/Slice_15:output:0(gradients/Reshape_15_grad/Shape:output:0*
T0*
_output_shapes
:@Ь
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:╡
$gradients/transpose_1_grad/transpose	Transpose'gradients/Reshape_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes

:	@Ь
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:╖
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes

:	@Ь
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:╖
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes

:	@Ь
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:╖
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes

:	@Ь
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:╖
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@@Ь
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:╖
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@@Ь
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:╖
$gradients/transpose_7_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@@Ь
,gradients/transpose_8_grad/InvertPermutationInvertPermutation=gradients_transpose_8_grad_invertpermutation_transpose_8_perm*
_output_shapes
:╖
$gradients/transpose_8_grad/transpose	Transpose)gradients/Reshape_7_grad/Reshape:output:00gradients/transpose_8_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@@ч
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0*gradients/Reshape_13_grad/Reshape:output:0*gradients/Reshape_14_grad/Reshape:output:0*gradients/Reshape_15_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:Ап
gradients/split_grad/concatConcatV2(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0*
_output_shapes
:		А╡
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0(gradients/transpose_7_grad/transpose:y:0(gradients/transpose_8_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0*
_output_shapes
:	@А\
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :Т
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: f
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:Аh
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:А╩
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0*
N* 
_output_shapes
::╥
gradients/concat_grad/SliceSlice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:А╓
gradients/concat_grad/Slice_1Slice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:А{
IdentityIdentity&gradients/transpose_grad/transpose:y:0*
T0*4
_output_shapes"
 :                  	t

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0*
T0*'
_output_shapes
:         @v

Identity_2Identity,gradients/ExpandDims_1_grad/Reshape:output:0*
T0*'
_output_shapes
:         @f

Identity_3Identity$gradients/split_grad/concat:output:0*
T0*
_output_shapes
:		Аh

Identity_4Identity&gradients/split_1_grad/concat:output:0*
T0*
_output_shapes
:	@Аd

Identity_5Identity&gradients/concat_grad/Slice_1:output:0*
T0*
_output_shapes	
:А"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*(
_construction_contextkEagerRuntime*И
_input_shapesЎ
є:         @:         @:         @:         @: :         @:         @:         @:                  @::                  	:         @:         @:АЦ::         @:         @: ::::::::: : : : *=
api_implements+)lstm_b4638b8b-6e40-4922-ab41-0ce5301502aa*
api_preferred_deviceGPU*A
forward_function_name(&__forward_gpu_lstm_with_fallback_22746*
go_backwards( *

time_major( :- )
'
_output_shapes
:         @:1-
+
_output_shapes
:         @:-)
'
_output_shapes
:         @:-)
'
_output_shapes
:         @:

_output_shapes
: :-)
'
_output_shapes
:         @:1-
+
_output_shapes
:         @:1-
+
_output_shapes
:         @::6
4
_output_shapes"
 :                  @:	

_output_shapes
:::
6
4
_output_shapes"
 :                  	:1-
+
_output_shapes
:         @:1-
+
_output_shapes
:         @:"

_output_shapes

:АЦ: 

_output_shapes
::-)
'
_output_shapes
:         @:-)
'
_output_shapes
:         @:

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Г┬
у
9__inference___backward_gpu_lstm_with_fallback_20066_20242
placeholder
placeholder_1
placeholder_2
placeholder_3
placeholder_43
/gradients_expanddims_2_grad_shape_strided_slice)
%gradients_squeeze_grad_shape_cudnnrnn+
'gradients_squeeze_1_grad_shape_cudnnrnn/
+gradients_strided_slice_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9
5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_15
1gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h,
(gradients_expanddims_1_grad_shape_init_c-
)gradients_concat_1_grad_mod_concat_1_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_permA
=gradients_transpose_7_grad_invertpermutation_transpose_7_permA
=gradients_transpose_8_grad_invertpermutation_transpose_8_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim)
%gradients_concat_grad_mod_concat_axis
identity

identity_1

identity_2

identity_3

identity_4

identity_5И^
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:         @d
gradients/grad_ys_1Identityplaceholder_1*
T0*+
_output_shapes
:         @`
gradients/grad_ys_2Identityplaceholder_2*
T0*'
_output_shapes
:         @`
gradients/grad_ys_3Identityplaceholder_3*
T0*'
_output_shapes
:         @O
gradients/grad_ys_4Identityplaceholder_4*
T0*
_output_shapes
: А
!gradients/ExpandDims_2_grad/ShapeShape/gradients_expanddims_2_grad_shape_strided_slice*
T0*
_output_shapes
:к
#gradients/ExpandDims_2_grad/ReshapeReshapegradients/grad_ys_1:output:0*gradients/ExpandDims_2_grad/Shape:output:0*
T0*'
_output_shapes
:         @q
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
:д
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*+
_output_shapes
:         @u
gradients/Squeeze_1_grad/ShapeShape'gradients_squeeze_1_grad_shape_cudnnrnn*
T0*
_output_shapes
:и
 gradients/Squeeze_1_grad/ReshapeReshapegradients/grad_ys_3:output:0'gradients/Squeeze_1_grad/Shape:output:0*
T0*+
_output_shapes
:         @┼
gradients/AddNAddNgradients/grad_ys_0:output:0,gradients/ExpandDims_2_grad/Reshape:output:0*
N*
T0*&
_class
loc:@gradients/grad_ys_0*'
_output_shapes
:         @}
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:Ж
3gradients/strided_slice_grad/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
         {
1gradients/strided_slice_grad/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 
5gradients/strided_slice_grad/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:в
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0<gradients/strided_slice_grad/StridedSliceGrad/begin:output:0:gradients/strided_slice_grad/StridedSliceGrad/end:output:0>gradients/strided_slice_grad/StridedSliceGrad/strides:output:0gradients/AddN:sum:0*
Index0*
T0*4
_output_shapes"
 :                  @*
shrink_axis_maskc
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
:м
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_11gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnn'gradients_squeeze_1_grad_shape_cudnnrnn6gradients/strided_slice_grad/StridedSliceGrad:output:0'gradients/Squeeze_grad/Reshape:output:0)gradients/Squeeze_1_grad/Reshape:output:0gradients_zeros_like_cudnnrnn*
T0*j
_output_shapesX
V:                  	:         @:         @:АЦЦ
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:┘
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*4
_output_shapes"
 :                  	u
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:┼
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*'
_output_shapes
:         @y
!gradients/ExpandDims_1_grad/ShapeShape(gradients_expanddims_1_grad_shape_init_c*
T0*
_output_shapes
:╔
#gradients/ExpandDims_1_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_c_backprop:0*gradients/ExpandDims_1_grad/Shape:output:0*
T0*'
_output_shapes
:         @^
gradients/concat_1_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :Ъ
gradients/concat_1_grad/modFloorMod)gradients_concat_1_grad_mod_concat_1_axis%gradients/concat_1_grad/Rank:output:0*
T0*
_output_shapes
: h
gradients/concat_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:└j
gradients/concat_1_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:└j
gradients/concat_1_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:└j
gradients/concat_1_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:└j
gradients/concat_1_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:А j
gradients/concat_1_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:А j
gradients/concat_1_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB:А j
gradients/concat_1_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB:А i
gradients/concat_1_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB:@i
gradients/concat_1_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB:@j
 gradients/concat_1_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB:@j
 gradients/concat_1_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB:@j
 gradients/concat_1_grad/Shape_12Const*
_output_shapes
:*
dtype0*
valueB:@j
 gradients/concat_1_grad/Shape_13Const*
_output_shapes
:*
dtype0*
valueB:@j
 gradients/concat_1_grad/Shape_14Const*
_output_shapes
:*
dtype0*
valueB:@j
 gradients/concat_1_grad/Shape_15Const*
_output_shapes
:*
dtype0*
valueB:@°
$gradients/concat_1_grad/ConcatOffsetConcatOffsetgradients/concat_1_grad/mod:z:0&gradients/concat_1_grad/Shape:output:0(gradients/concat_1_grad/Shape_1:output:0(gradients/concat_1_grad/Shape_2:output:0(gradients/concat_1_grad/Shape_3:output:0(gradients/concat_1_grad/Shape_4:output:0(gradients/concat_1_grad/Shape_5:output:0(gradients/concat_1_grad/Shape_6:output:0(gradients/concat_1_grad/Shape_7:output:0(gradients/concat_1_grad/Shape_8:output:0(gradients/concat_1_grad/Shape_9:output:0)gradients/concat_1_grad/Shape_10:output:0)gradients/concat_1_grad/Shape_11:output:0)gradients/concat_1_grad/Shape_12:output:0)gradients/concat_1_grad/Shape_13:output:0)gradients/concat_1_grad/Shape_14:output:0)gradients/concat_1_grad/Shape_15:output:0*
N*t
_output_shapesb
`::::::::::::::::ь
gradients/concat_1_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:0&gradients/concat_1_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:└Ё
gradients/concat_1_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:1(gradients/concat_1_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:└Ё
gradients/concat_1_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:2(gradients/concat_1_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:└Ё
gradients/concat_1_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:3(gradients/concat_1_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:└Ё
gradients/concat_1_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:4(gradients/concat_1_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:А Ё
gradients/concat_1_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:5(gradients/concat_1_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:А Ё
gradients/concat_1_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:6(gradients/concat_1_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes	
:А Ё
gradients/concat_1_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:7(gradients/concat_1_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes	
:А я
gradients/concat_1_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:8(gradients/concat_1_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
:@я
gradients/concat_1_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:9(gradients/concat_1_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
:@Є
 gradients/concat_1_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:10)gradients/concat_1_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
:@Є
 gradients/concat_1_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:11)gradients/concat_1_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
:@Є
 gradients/concat_1_grad/Slice_12Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:12)gradients/concat_1_grad/Shape_12:output:0*
Index0*
T0*
_output_shapes
:@Є
 gradients/concat_1_grad/Slice_13Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:13)gradients/concat_1_grad/Shape_13:output:0*
Index0*
T0*
_output_shapes
:@Є
 gradients/concat_1_grad/Slice_14Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:14)gradients/concat_1_grad/Shape_14:output:0*
Index0*
T0*
_output_shapes
:@Є
 gradients/concat_1_grad/Slice_15Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:15)gradients/concat_1_grad/Shape_15:output:0*
Index0*
T0*
_output_shapes
:@m
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"@   	   б
gradients/Reshape_grad/ReshapeReshape&gradients/concat_1_grad/Slice:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes

:@	o
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"@   	   з
 gradients/Reshape_1_grad/ReshapeReshape(gradients/concat_1_grad/Slice_1:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes

:@	o
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"@   	   з
 gradients/Reshape_2_grad/ReshapeReshape(gradients/concat_1_grad/Slice_2:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes

:@	o
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"@   	   з
 gradients/Reshape_3_grad/ReshapeReshape(gradients/concat_1_grad/Slice_3:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes

:@	o
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"@   @   з
 gradients/Reshape_4_grad/ReshapeReshape(gradients/concat_1_grad/Slice_4:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:@@o
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"@   @   з
 gradients/Reshape_5_grad/ReshapeReshape(gradients/concat_1_grad/Slice_5:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:@@o
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"@   @   з
 gradients/Reshape_6_grad/ReshapeReshape(gradients/concat_1_grad/Slice_6:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:@@o
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"@   @   з
 gradients/Reshape_7_grad/ReshapeReshape(gradients/concat_1_grad/Slice_7:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes

:@@h
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:@г
 gradients/Reshape_8_grad/ReshapeReshape(gradients/concat_1_grad/Slice_8:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
:@h
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:@г
 gradients/Reshape_9_grad/ReshapeReshape(gradients/concat_1_grad/Slice_9:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
:@i
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:@ж
!gradients/Reshape_10_grad/ReshapeReshape)gradients/concat_1_grad/Slice_10:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
:@i
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:@ж
!gradients/Reshape_11_grad/ReshapeReshape)gradients/concat_1_grad/Slice_11:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
:@i
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:@ж
!gradients/Reshape_12_grad/ReshapeReshape)gradients/concat_1_grad/Slice_12:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
:@i
gradients/Reshape_13_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:@ж
!gradients/Reshape_13_grad/ReshapeReshape)gradients/concat_1_grad/Slice_13:output:0(gradients/Reshape_13_grad/Shape:output:0*
T0*
_output_shapes
:@i
gradients/Reshape_14_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:@ж
!gradients/Reshape_14_grad/ReshapeReshape)gradients/concat_1_grad/Slice_14:output:0(gradients/Reshape_14_grad/Shape:output:0*
T0*
_output_shapes
:@i
gradients/Reshape_15_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:@ж
!gradients/Reshape_15_grad/ReshapeReshape)gradients/concat_1_grad/Slice_15:output:0(gradients/Reshape_15_grad/Shape:output:0*
T0*
_output_shapes
:@Ь
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:╡
$gradients/transpose_1_grad/transpose	Transpose'gradients/Reshape_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes

:	@Ь
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:╖
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes

:	@Ь
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:╖
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes

:	@Ь
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:╖
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes

:	@Ь
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:╖
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@@Ь
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:╖
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@@Ь
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:╖
$gradients/transpose_7_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@@Ь
,gradients/transpose_8_grad/InvertPermutationInvertPermutation=gradients_transpose_8_grad_invertpermutation_transpose_8_perm*
_output_shapes
:╖
$gradients/transpose_8_grad/transpose	Transpose)gradients/Reshape_7_grad/Reshape:output:00gradients/transpose_8_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@@ч
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0*gradients/Reshape_13_grad/Reshape:output:0*gradients/Reshape_14_grad/Reshape:output:0*gradients/Reshape_15_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:Ап
gradients/split_grad/concatConcatV2(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0*
_output_shapes
:		А╡
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0(gradients/transpose_7_grad/transpose:y:0(gradients/transpose_8_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0*
_output_shapes
:	@А\
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :Т
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: f
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:Аh
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:А╩
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0*
N* 
_output_shapes
::╥
gradients/concat_grad/SliceSlice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:А╓
gradients/concat_grad/Slice_1Slice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:А{
IdentityIdentity&gradients/transpose_grad/transpose:y:0*
T0*4
_output_shapes"
 :                  	t

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0*
T0*'
_output_shapes
:         @v

Identity_2Identity,gradients/ExpandDims_1_grad/Reshape:output:0*
T0*'
_output_shapes
:         @f

Identity_3Identity$gradients/split_grad/concat:output:0*
T0*
_output_shapes
:		Аh

Identity_4Identity&gradients/split_1_grad/concat:output:0*
T0*
_output_shapes
:	@Аd

Identity_5Identity&gradients/concat_grad/Slice_1:output:0*
T0*
_output_shapes	
:А"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*(
_construction_contextkEagerRuntime*И
_input_shapesЎ
є:         @:         @:         @:         @: :         @:         @:         @:                  @::                  	:         @:         @:АЦ::         @:         @: ::::::::: : : : *=
api_implements+)lstm_59b087b6-9d64-470e-a94e-2d14718ac25a*
api_preferred_deviceGPU*A
forward_function_name(&__forward_gpu_lstm_with_fallback_20241*
go_backwards( *

time_major( :- )
'
_output_shapes
:         @:1-
+
_output_shapes
:         @:-)
'
_output_shapes
:         @:-)
'
_output_shapes
:         @:

_output_shapes
: :-)
'
_output_shapes
:         @:1-
+
_output_shapes
:         @:1-
+
_output_shapes
:         @::6
4
_output_shapes"
 :                  @:	

_output_shapes
:::
6
4
_output_shapes"
 :                  	:1-
+
_output_shapes
:         @:1-
+
_output_shapes
:         @:"

_output_shapes

:АЦ: 

_output_shapes
::-)
'
_output_shapes
:         @:-)
'
_output_shapes
:         @:

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
К2
ж
 __inference__wrapped_model_20250
normalization_input"
sequential_normalization_sub_y#
sequential_normalization_sqrt_x?
,sequential_lstm_read_readvariableop_resource:		АA
.sequential_lstm_read_1_readvariableop_resource:	@А=
.sequential_lstm_read_2_readvariableop_resource:	АB
/sequential_dense_matmul_readvariableop_resource:	@э?
0sequential_dense_biasadd_readvariableop_resource:	э
identityИв'sequential/dense/BiasAdd/ReadVariableOpв&sequential/dense/MatMul/ReadVariableOpв#sequential/lstm/Read/ReadVariableOpв%sequential/lstm/Read_1/ReadVariableOpв%sequential/lstm/Read_2/ReadVariableOpЧ
sequential/normalization/subSubnormalization_inputsequential_normalization_sub_y*
T0*4
_output_shapes"
 :                  	s
sequential/normalization/SqrtSqrtsequential_normalization_sqrt_x*
T0*"
_output_shapes
:	g
"sequential/normalization/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *Х┐╓3и
 sequential/normalization/MaximumMaximum!sequential/normalization/Sqrt:y:0+sequential/normalization/Maximum/y:output:0*
T0*"
_output_shapes
:	▓
 sequential/normalization/truedivRealDiv sequential/normalization/sub:z:0$sequential/normalization/Maximum:z:0*
T0*4
_output_shapes"
 :                  	i
sequential/lstm/ShapeShape$sequential/normalization/truediv:z:0*
T0*
_output_shapes
:m
#sequential/lstm/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%sequential/lstm/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%sequential/lstm/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
sequential/lstm/strided_sliceStridedSlicesequential/lstm/Shape:output:0,sequential/lstm/strided_slice/stack:output:0.sequential/lstm/strided_slice/stack_1:output:0.sequential/lstm/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask`
sequential/lstm/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@г
sequential/lstm/zeros/packedPack&sequential/lstm/strided_slice:output:0'sequential/lstm/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:`
sequential/lstm/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Ь
sequential/lstm/zerosFill%sequential/lstm/zeros/packed:output:0$sequential/lstm/zeros/Const:output:0*
T0*'
_output_shapes
:         @b
 sequential/lstm/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@з
sequential/lstm/zeros_1/packedPack&sequential/lstm/strided_slice:output:0)sequential/lstm/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:b
sequential/lstm/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    в
sequential/lstm/zeros_1Fill'sequential/lstm/zeros_1/packed:output:0&sequential/lstm/zeros_1/Const:output:0*
T0*'
_output_shapes
:         @С
#sequential/lstm/Read/ReadVariableOpReadVariableOp,sequential_lstm_read_readvariableop_resource*
_output_shapes
:		А*
dtype0{
sequential/lstm/IdentityIdentity+sequential/lstm/Read/ReadVariableOp:value:0*
T0*
_output_shapes
:		АХ
%sequential/lstm/Read_1/ReadVariableOpReadVariableOp.sequential_lstm_read_1_readvariableop_resource*
_output_shapes
:	@А*
dtype0
sequential/lstm/Identity_1Identity-sequential/lstm/Read_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	@АС
%sequential/lstm/Read_2/ReadVariableOpReadVariableOp.sequential_lstm_read_2_readvariableop_resource*
_output_shapes	
:А*
dtype0{
sequential/lstm/Identity_2Identity-sequential/lstm/Read_2/ReadVariableOp:value:0*
T0*
_output_shapes	
:Ап
sequential/lstm/PartitionedCallPartitionedCall$sequential/normalization/truediv:z:0sequential/lstm/zeros:output:0 sequential/lstm/zeros_1:output:0!sequential/lstm/Identity:output:0#sequential/lstm/Identity_1:output:0#sequential/lstm/Identity_2:output:0*
Tin

2*
Tout	
2*
_collective_manager_ids
 *f
_output_shapesT
R:         @:         @:         @:         @: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *(
f#R!
__inference_standard_lstm_19971Ч
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource*
_output_shapes
:	@э*
dtype0о
sequential/dense/MatMulMatMul(sequential/lstm/PartitionedCall:output:0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         эХ
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes	
:э*
dtype0к
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         эq
IdentityIdentity!sequential/dense/BiasAdd:output:0^NoOp*
T0*(
_output_shapes
:         эП
NoOpNoOp(^sequential/dense/BiasAdd/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp$^sequential/lstm/Read/ReadVariableOp&^sequential/lstm/Read_1/ReadVariableOp&^sequential/lstm/Read_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:'                           :	:	: : : : : 2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp2J
#sequential/lstm/Read/ReadVariableOp#sequential/lstm/Read/ReadVariableOp2N
%sequential/lstm/Read_1/ReadVariableOp%sequential/lstm/Read_1/ReadVariableOp2N
%sequential/lstm/Read_2/ReadVariableOp%sequential/lstm/Read_2/ReadVariableOp:r n
=
_output_shapes+
):'                           
-
_user_specified_namenormalization_input:($
"
_output_shapes
:	:($
"
_output_shapes
:	
╥
°
E__inference_sequential_layer_call_and_return_conditional_losses_22203
normalization_input
normalization_sub_y
normalization_sqrt_x

lstm_22190:		А

lstm_22192:	@А

lstm_22194:	А
dense_22197:	@э
dense_22199:	э
identityИвdense/StatefulPartitionedCallвlstm/StatefulPartitionedCallБ
normalization/subSubnormalization_inputnormalization_sub_y*
T0*4
_output_shapes"
 :                  	]
normalization/SqrtSqrtnormalization_sqrt_x*
T0*"
_output_shapes
:	\
normalization/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *Х┐╓3З
normalization/MaximumMaximumnormalization/Sqrt:y:0 normalization/Maximum/y:output:0*
T0*"
_output_shapes
:	С
normalization/truedivRealDivnormalization/sub:z:0normalization/Maximum:z:0*
T0*4
_output_shapes"
 :                  	■
lstm/StatefulPartitionedCallStatefulPartitionedCallnormalization/truediv:z:0
lstm_22190
lstm_22192
lstm_22194*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *H
fCRA
?__inference_lstm_layer_call_and_return_conditional_losses_22068Б
dense/StatefulPartitionedCallStatefulPartitionedCall%lstm/StatefulPartitionedCall:output:0dense_22197dense_22199*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         э*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_21592v
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         эЕ
NoOpNoOp^dense/StatefulPartitionedCall^lstm/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:'                           :	:	: : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2<
lstm/StatefulPartitionedCalllstm/StatefulPartitionedCall:r n
=
_output_shapes+
):'                           
-
_user_specified_namenormalization_input:($
"
_output_shapes
:	:($
"
_output_shapes
:	
╪)
╬
while_body_20764
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_bias_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel#
while_matmul_1_recurrent_kernel
while_biasadd_biasИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    	   ж
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         	*
element_dtype0Т
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*(
_output_shapes
:         АГ
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*(
_output_shapes
:         Аw
	while/addAddV2while/MatMul:product:0while/MatMul_1:product:0*
T0*(
_output_shapes
:         Аp
while/BiasAddBiasAddwhile/add:z:0while_biasadd_bias_0*
T0*(
_output_shapes
:         АW
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :╚
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*`
_output_shapesN
L:         @:         @:         @:         @*
	num_split`
while/SigmoidSigmoidwhile/split:output:0*
T0*'
_output_shapes
:         @b
while/Sigmoid_1Sigmoidwhile/split:output:1*
T0*'
_output_shapes
:         @l
	while/mulMulwhile/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:         @Z

while/TanhTanhwhile/split:output:2*
T0*'
_output_shapes
:         @g
while/mul_1Mulwhile/Sigmoid:y:0while/Tanh:y:0*
T0*'
_output_shapes
:         @f
while/add_1AddV2while/mul:z:0while/mul_1:z:0*
T0*'
_output_shapes
:         @b
while/Sigmoid_2Sigmoidwhile/split:output:3*
T0*'
_output_shapes
:         @W
while/Tanh_1Tanhwhile/add_1:z:0*
T0*'
_output_shapes
:         @k
while/mul_2Mulwhile/Sigmoid_2:y:0while/Tanh_1:y:0*
T0*'
_output_shapes
:         @r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : р
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/mul_2:z:0*
_output_shapes
: *
element_dtype0:щш╥O
while/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :`
while/add_2AddV2while_placeholderwhile/add_2/y:output:0*
T0*
_output_shapes
: O
while/add_3/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_3AddV2while_while_loop_counterwhile/add_3/y:output:0*
T0*
_output_shapes
: L
while/IdentityIdentitywhile/add_3:z:0*
T0*
_output_shapes
: ]
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: N
while/Identity_2Identitywhile/add_2:z:0*
T0*
_output_shapes
: y
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: _
while/Identity_4Identitywhile/mul_2:z:0*
T0*'
_output_shapes
:         @_
while/Identity_5Identitywhile/add_1:z:0*
T0*'
_output_shapes
:         @"*
while_biasadd_biaswhile_biasadd_bias_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O: : : : :         @:         @: : :		А:	@А:А: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         @:-)
'
_output_shapes
:         @:

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:		А:%	!

_output_shapes
:	@А:!


_output_shapes	
:А
М	
╝
while_cond_22831
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice3
/while_while_cond_22831___redundant_placeholder03
/while_while_cond_22831___redundant_placeholder13
/while_while_cond_22831___redundant_placeholder23
/while_while_cond_22831___redundant_placeholder3
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :         @:         @: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         @:-)
'
_output_shapes
:         @:

_output_shapes
: :

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
╪)
╬
while_body_21709
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_bias_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel#
while_matmul_1_recurrent_kernel
while_biasadd_biasИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    	   ж
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         	*
element_dtype0Т
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*(
_output_shapes
:         АГ
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*(
_output_shapes
:         Аw
	while/addAddV2while/MatMul:product:0while/MatMul_1:product:0*
T0*(
_output_shapes
:         Аp
while/BiasAddBiasAddwhile/add:z:0while_biasadd_bias_0*
T0*(
_output_shapes
:         АW
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :╚
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*`
_output_shapesN
L:         @:         @:         @:         @*
	num_split`
while/SigmoidSigmoidwhile/split:output:0*
T0*'
_output_shapes
:         @b
while/Sigmoid_1Sigmoidwhile/split:output:1*
T0*'
_output_shapes
:         @l
	while/mulMulwhile/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:         @Z

while/TanhTanhwhile/split:output:2*
T0*'
_output_shapes
:         @g
while/mul_1Mulwhile/Sigmoid:y:0while/Tanh:y:0*
T0*'
_output_shapes
:         @f
while/add_1AddV2while/mul:z:0while/mul_1:z:0*
T0*'
_output_shapes
:         @b
while/Sigmoid_2Sigmoidwhile/split:output:3*
T0*'
_output_shapes
:         @W
while/Tanh_1Tanhwhile/add_1:z:0*
T0*'
_output_shapes
:         @k
while/mul_2Mulwhile/Sigmoid_2:y:0while/Tanh_1:y:0*
T0*'
_output_shapes
:         @r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : р
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/mul_2:z:0*
_output_shapes
: *
element_dtype0:щш╥O
while/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :`
while/add_2AddV2while_placeholderwhile/add_2/y:output:0*
T0*
_output_shapes
: O
while/add_3/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_3AddV2while_while_loop_counterwhile/add_3/y:output:0*
T0*
_output_shapes
: L
while/IdentityIdentitywhile/add_3:z:0*
T0*
_output_shapes
: ]
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: N
while/Identity_2Identitywhile/add_2:z:0*
T0*
_output_shapes
: y
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: _
while/Identity_4Identitywhile/mul_2:z:0*
T0*'
_output_shapes
:         @_
while/Identity_5Identitywhile/add_1:z:0*
T0*'
_output_shapes
:         @"*
while_biasadd_biaswhile_biasadd_bias_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O: : : : :         @:         @: : :		А:	@А:А: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         @:-)
'
_output_shapes
:         @:

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:		А:%	!

_output_shapes
:	@А:!


_output_shapes	
:А
М	
╝
while_cond_22389
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice3
/while_while_cond_22389___redundant_placeholder03
/while_while_cond_22389___redundant_placeholder13
/while_while_cond_22389___redundant_placeholder23
/while_while_cond_22389___redundant_placeholder3
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :         @:         @: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         @:-)
'
_output_shapes
:         @:

_output_shapes
: :

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
Г┬
у
9__inference___backward_gpu_lstm_with_fallback_20505_20681
placeholder
placeholder_1
placeholder_2
placeholder_3
placeholder_43
/gradients_expanddims_2_grad_shape_strided_slice)
%gradients_squeeze_grad_shape_cudnnrnn+
'gradients_squeeze_1_grad_shape_cudnnrnn/
+gradients_strided_slice_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9
5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_15
1gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h,
(gradients_expanddims_1_grad_shape_init_c-
)gradients_concat_1_grad_mod_concat_1_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_permA
=gradients_transpose_7_grad_invertpermutation_transpose_7_permA
=gradients_transpose_8_grad_invertpermutation_transpose_8_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim)
%gradients_concat_grad_mod_concat_axis
identity

identity_1

identity_2

identity_3

identity_4

identity_5И^
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:         @d
gradients/grad_ys_1Identityplaceholder_1*
T0*+
_output_shapes
:         @`
gradients/grad_ys_2Identityplaceholder_2*
T0*'
_output_shapes
:         @`
gradients/grad_ys_3Identityplaceholder_3*
T0*'
_output_shapes
:         @O
gradients/grad_ys_4Identityplaceholder_4*
T0*
_output_shapes
: А
!gradients/ExpandDims_2_grad/ShapeShape/gradients_expanddims_2_grad_shape_strided_slice*
T0*
_output_shapes
:к
#gradients/ExpandDims_2_grad/ReshapeReshapegradients/grad_ys_1:output:0*gradients/ExpandDims_2_grad/Shape:output:0*
T0*'
_output_shapes
:         @q
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
:д
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*+
_output_shapes
:         @u
gradients/Squeeze_1_grad/ShapeShape'gradients_squeeze_1_grad_shape_cudnnrnn*
T0*
_output_shapes
:и
 gradients/Squeeze_1_grad/ReshapeReshapegradients/grad_ys_3:output:0'gradients/Squeeze_1_grad/Shape:output:0*
T0*+
_output_shapes
:         @┼
gradients/AddNAddNgradients/grad_ys_0:output:0,gradients/ExpandDims_2_grad/Reshape:output:0*
N*
T0*&
_class
loc:@gradients/grad_ys_0*'
_output_shapes
:         @}
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:Ж
3gradients/strided_slice_grad/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
         {
1gradients/strided_slice_grad/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 
5gradients/strided_slice_grad/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:в
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0<gradients/strided_slice_grad/StridedSliceGrad/begin:output:0:gradients/strided_slice_grad/StridedSliceGrad/end:output:0>gradients/strided_slice_grad/StridedSliceGrad/strides:output:0gradients/AddN:sum:0*
Index0*
T0*4
_output_shapes"
 :                  @*
shrink_axis_maskc
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
:м
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_11gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnn'gradients_squeeze_1_grad_shape_cudnnrnn6gradients/strided_slice_grad/StridedSliceGrad:output:0'gradients/Squeeze_grad/Reshape:output:0)gradients/Squeeze_1_grad/Reshape:output:0gradients_zeros_like_cudnnrnn*
T0*j
_output_shapesX
V:                  	:         @:         @:АЦЦ
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:┘
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*4
_output_shapes"
 :                  	u
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:┼
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*'
_output_shapes
:         @y
!gradients/ExpandDims_1_grad/ShapeShape(gradients_expanddims_1_grad_shape_init_c*
T0*
_output_shapes
:╔
#gradients/ExpandDims_1_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_c_backprop:0*gradients/ExpandDims_1_grad/Shape:output:0*
T0*'
_output_shapes
:         @^
gradients/concat_1_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :Ъ
gradients/concat_1_grad/modFloorMod)gradients_concat_1_grad_mod_concat_1_axis%gradients/concat_1_grad/Rank:output:0*
T0*
_output_shapes
: h
gradients/concat_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:└j
gradients/concat_1_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:└j
gradients/concat_1_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:└j
gradients/concat_1_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:└j
gradients/concat_1_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:А j
gradients/concat_1_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:А j
gradients/concat_1_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB:А j
gradients/concat_1_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB:А i
gradients/concat_1_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB:@i
gradients/concat_1_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB:@j
 gradients/concat_1_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB:@j
 gradients/concat_1_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB:@j
 gradients/concat_1_grad/Shape_12Const*
_output_shapes
:*
dtype0*
valueB:@j
 gradients/concat_1_grad/Shape_13Const*
_output_shapes
:*
dtype0*
valueB:@j
 gradients/concat_1_grad/Shape_14Const*
_output_shapes
:*
dtype0*
valueB:@j
 gradients/concat_1_grad/Shape_15Const*
_output_shapes
:*
dtype0*
valueB:@°
$gradients/concat_1_grad/ConcatOffsetConcatOffsetgradients/concat_1_grad/mod:z:0&gradients/concat_1_grad/Shape:output:0(gradients/concat_1_grad/Shape_1:output:0(gradients/concat_1_grad/Shape_2:output:0(gradients/concat_1_grad/Shape_3:output:0(gradients/concat_1_grad/Shape_4:output:0(gradients/concat_1_grad/Shape_5:output:0(gradients/concat_1_grad/Shape_6:output:0(gradients/concat_1_grad/Shape_7:output:0(gradients/concat_1_grad/Shape_8:output:0(gradients/concat_1_grad/Shape_9:output:0)gradients/concat_1_grad/Shape_10:output:0)gradients/concat_1_grad/Shape_11:output:0)gradients/concat_1_grad/Shape_12:output:0)gradients/concat_1_grad/Shape_13:output:0)gradients/concat_1_grad/Shape_14:output:0)gradients/concat_1_grad/Shape_15:output:0*
N*t
_output_shapesb
`::::::::::::::::ь
gradients/concat_1_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:0&gradients/concat_1_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:└Ё
gradients/concat_1_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:1(gradients/concat_1_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:└Ё
gradients/concat_1_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:2(gradients/concat_1_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:└Ё
gradients/concat_1_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:3(gradients/concat_1_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:└Ё
gradients/concat_1_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:4(gradients/concat_1_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:А Ё
gradients/concat_1_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:5(gradients/concat_1_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:А Ё
gradients/concat_1_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:6(gradients/concat_1_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes	
:А Ё
gradients/concat_1_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:7(gradients/concat_1_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes	
:А я
gradients/concat_1_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:8(gradients/concat_1_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
:@я
gradients/concat_1_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:9(gradients/concat_1_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
:@Є
 gradients/concat_1_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:10)gradients/concat_1_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
:@Є
 gradients/concat_1_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:11)gradients/concat_1_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
:@Є
 gradients/concat_1_grad/Slice_12Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:12)gradients/concat_1_grad/Shape_12:output:0*
Index0*
T0*
_output_shapes
:@Є
 gradients/concat_1_grad/Slice_13Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:13)gradients/concat_1_grad/Shape_13:output:0*
Index0*
T0*
_output_shapes
:@Є
 gradients/concat_1_grad/Slice_14Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:14)gradients/concat_1_grad/Shape_14:output:0*
Index0*
T0*
_output_shapes
:@Є
 gradients/concat_1_grad/Slice_15Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:15)gradients/concat_1_grad/Shape_15:output:0*
Index0*
T0*
_output_shapes
:@m
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"@   	   б
gradients/Reshape_grad/ReshapeReshape&gradients/concat_1_grad/Slice:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes

:@	o
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"@   	   з
 gradients/Reshape_1_grad/ReshapeReshape(gradients/concat_1_grad/Slice_1:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes

:@	o
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"@   	   з
 gradients/Reshape_2_grad/ReshapeReshape(gradients/concat_1_grad/Slice_2:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes

:@	o
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"@   	   з
 gradients/Reshape_3_grad/ReshapeReshape(gradients/concat_1_grad/Slice_3:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes

:@	o
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"@   @   з
 gradients/Reshape_4_grad/ReshapeReshape(gradients/concat_1_grad/Slice_4:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:@@o
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"@   @   з
 gradients/Reshape_5_grad/ReshapeReshape(gradients/concat_1_grad/Slice_5:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:@@o
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"@   @   з
 gradients/Reshape_6_grad/ReshapeReshape(gradients/concat_1_grad/Slice_6:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:@@o
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"@   @   з
 gradients/Reshape_7_grad/ReshapeReshape(gradients/concat_1_grad/Slice_7:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes

:@@h
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:@г
 gradients/Reshape_8_grad/ReshapeReshape(gradients/concat_1_grad/Slice_8:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
:@h
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:@г
 gradients/Reshape_9_grad/ReshapeReshape(gradients/concat_1_grad/Slice_9:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
:@i
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:@ж
!gradients/Reshape_10_grad/ReshapeReshape)gradients/concat_1_grad/Slice_10:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
:@i
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:@ж
!gradients/Reshape_11_grad/ReshapeReshape)gradients/concat_1_grad/Slice_11:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
:@i
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:@ж
!gradients/Reshape_12_grad/ReshapeReshape)gradients/concat_1_grad/Slice_12:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
:@i
gradients/Reshape_13_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:@ж
!gradients/Reshape_13_grad/ReshapeReshape)gradients/concat_1_grad/Slice_13:output:0(gradients/Reshape_13_grad/Shape:output:0*
T0*
_output_shapes
:@i
gradients/Reshape_14_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:@ж
!gradients/Reshape_14_grad/ReshapeReshape)gradients/concat_1_grad/Slice_14:output:0(gradients/Reshape_14_grad/Shape:output:0*
T0*
_output_shapes
:@i
gradients/Reshape_15_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:@ж
!gradients/Reshape_15_grad/ReshapeReshape)gradients/concat_1_grad/Slice_15:output:0(gradients/Reshape_15_grad/Shape:output:0*
T0*
_output_shapes
:@Ь
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:╡
$gradients/transpose_1_grad/transpose	Transpose'gradients/Reshape_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes

:	@Ь
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:╖
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes

:	@Ь
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:╖
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes

:	@Ь
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:╖
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes

:	@Ь
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:╖
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@@Ь
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:╖
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@@Ь
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:╖
$gradients/transpose_7_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@@Ь
,gradients/transpose_8_grad/InvertPermutationInvertPermutation=gradients_transpose_8_grad_invertpermutation_transpose_8_perm*
_output_shapes
:╖
$gradients/transpose_8_grad/transpose	Transpose)gradients/Reshape_7_grad/Reshape:output:00gradients/transpose_8_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@@ч
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0*gradients/Reshape_13_grad/Reshape:output:0*gradients/Reshape_14_grad/Reshape:output:0*gradients/Reshape_15_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:Ап
gradients/split_grad/concatConcatV2(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0*
_output_shapes
:		А╡
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0(gradients/transpose_7_grad/transpose:y:0(gradients/transpose_8_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0*
_output_shapes
:	@А\
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :Т
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: f
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:Аh
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:А╩
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0*
N* 
_output_shapes
::╥
gradients/concat_grad/SliceSlice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:А╓
gradients/concat_grad/Slice_1Slice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:А{
IdentityIdentity&gradients/transpose_grad/transpose:y:0*
T0*4
_output_shapes"
 :                  	t

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0*
T0*'
_output_shapes
:         @v

Identity_2Identity,gradients/ExpandDims_1_grad/Reshape:output:0*
T0*'
_output_shapes
:         @f

Identity_3Identity$gradients/split_grad/concat:output:0*
T0*
_output_shapes
:		Аh

Identity_4Identity&gradients/split_1_grad/concat:output:0*
T0*
_output_shapes
:	@Аd

Identity_5Identity&gradients/concat_grad/Slice_1:output:0*
T0*
_output_shapes	
:А"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*(
_construction_contextkEagerRuntime*И
_input_shapesЎ
є:         @:         @:         @:         @: :         @:         @:         @:                  @::                  	:         @:         @:АЦ::         @:         @: ::::::::: : : : *=
api_implements+)lstm_8d97cef0-5325-4590-9604-659863d02e30*
api_preferred_deviceGPU*A
forward_function_name(&__forward_gpu_lstm_with_fallback_20680*
go_backwards( *

time_major( :- )
'
_output_shapes
:         @:1-
+
_output_shapes
:         @:-)
'
_output_shapes
:         @:-)
'
_output_shapes
:         @:

_output_shapes
: :-)
'
_output_shapes
:         @:1-
+
_output_shapes
:         @:1-
+
_output_shapes
:         @::6
4
_output_shapes"
 :                  @:	

_output_shapes
:::
6
4
_output_shapes"
 :                  	:1-
+
_output_shapes
:         @:1-
+
_output_shapes
:         @:"

_output_shapes

:АЦ: 

_output_shapes
::-)
'
_output_shapes
:         @:-)
'
_output_shapes
:         @:

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
╟@
╦
(__inference_gpu_lstm_with_fallback_20065

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4Иc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :                  	P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : o

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*+
_output_shapes
:         @R
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : s
ExpandDims_1
ExpandDimsinit_cExpandDims_1/dim:output:0*
T0*+
_output_shapes
:         @Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :И
splitSplitsplit/split_dim:output:0kernel*
T0*<
_output_shapes*
(:	@:	@:	@:	@*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ц
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*<
_output_shapes*
(:@@:@@:@@:@@*
	num_splitY

zeros_likeConst*
_output_shapes	
:А*
dtype0*
valueBА*    M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : r
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes	
:АS
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Э
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*D
_output_shapes2
0:@:@:@:@:@:@:@:@*
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
         a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes

:@	Y
ReshapeReshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:└a
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes

:@	[
	Reshape_1Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:└a
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

:@	[
	Reshape_2Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:└a
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes

:@	[
	Reshape_3Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:└a
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:@@[
	Reshape_4Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:А a
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0*
_output_shapes

:@@[
	Reshape_5Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:А a
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0*
_output_shapes

:@@[
	Reshape_6Reshapetranspose_7:y:0Const:output:0*
T0*
_output_shapes	
:А a
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0*
_output_shapes

:@@[
	Reshape_7Reshapetranspose_8:y:0Const:output:0*
T0*
_output_shapes	
:А [
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
:@[
	Reshape_9Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
:@\

Reshape_10Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
:@\

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
:@\

Reshape_12Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
:@\

Reshape_13Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
:@\

Reshape_14Reshapesplit_2:output:6Const:output:0*
T0*
_output_shapes
:@\

Reshape_15Reshapesplit_2:output:7Const:output:0*
T0*
_output_shapes
:@O
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : а
concat_1ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes

:АЦ╙
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1:output:0*
T0*f
_output_shapesT
R:                  @:         @:         @:f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
         _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:х
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         @*
shrink_axis_maskp
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:         @*
squeeze_dims
 r
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*'
_output_shapes
:         @*
squeeze_dims
 R
ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B :Г
ExpandDims_2
ExpandDimsstrided_slice:output:0ExpandDims_2/dim:output:0*
T0*+
_output_shapes
:         @[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @^
IdentityIdentitystrided_slice:output:0*
T0*'
_output_shapes
:         @c

Identity_1IdentityExpandDims_2:output:0*
T0*+
_output_shapes
:         @Z

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:         @\

Identity_3IdentitySqueeze_1:output:0*
T0*'
_output_shapes
:         @I

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:                  	:         @:         @:		А:	@А:А*=
api_implements+)lstm_59b087b6-9d64-470e-a94e-2d14718ac25a*
api_preferred_deviceGPU*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :                  	
 
_user_specified_nameinputs:OK
'
_output_shapes
:         @
 
_user_specified_nameinit_h:OK
'
_output_shapes
:         @
 
_user_specified_nameinit_c:GC

_output_shapes
:		А
 
_user_specified_namekernel:QM

_output_shapes
:	@А
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:А

_user_specified_namebias
╗;
┐
__inference_standard_lstm_22918

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :                  	B
ShapeShapetranspose:y:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ▓
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    	   р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:щ
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         	*
shrink_axis_maske
MatMulMatMulstrided_slice_1:output:0kernel*
T0*(
_output_shapes
:         А_
MatMul_1MatMulinit_hrecurrent_kernel*
T0*(
_output_shapes
:         Аe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         АT
BiasAddBiasAddadd:z:0bias*
T0*(
_output_shapes
:         АQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :╢
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:         @:         @:         @:         @*
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         @V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         @S
mulMulSigmoid_1:y:0init_c*
T0*'
_output_shapes
:         @N
TanhTanhsplit:output:2*
T0*'
_output_shapes
:         @U
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:         @T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         @V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         @K
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:         @Y
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:         @n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :┼
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ┤
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hinit_cstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelrecurrent_kernelbias*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*c
_output_shapesQ
O: : : : :         @:         @: : :		А:	@А:А* 
_read_only_resource_inputs
 *
_stateful_parallelism( *
bodyR
while_body_22832*
condR
while_cond_22831*b
output_shapesQ
O: : : : :         @:         @: : :		А:	@А:А*
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   ╓
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         @*
element_dtype0*
num_elementsh
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:З
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         @*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ц
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:         @[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  А?`
IdentityIdentitystrided_slice_2:output:0*
T0*'
_output_shapes
:         @]

Identity_1Identitytranspose_1:y:0*
T0*+
_output_shapes
:         @X

Identity_2Identitywhile:output:4*
T0*'
_output_shapes
:         @X

Identity_3Identitywhile:output:5*
T0*'
_output_shapes
:         @I

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:                  	:         @:         @:		А:	@А:А*=
api_implements+)lstm_6046d01b-8f35-430c-8827-9bde89cc14a6*
api_preferred_deviceCPU*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :                  	
 
_user_specified_nameinputs:OK
'
_output_shapes
:         @
 
_user_specified_nameinit_h:OK
'
_output_shapes
:         @
 
_user_specified_nameinit_c:GC

_output_shapes
:		А
 
_user_specified_namekernel:QM

_output_shapes
:	@А
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:А

_user_specified_namebias
в
╣
?__inference_lstm_layer_call_and_return_conditional_losses_24957

inputs/
read_readvariableop_resource:		А1
read_1_readvariableop_resource:	@А-
read_2_readvariableop_resource:	А

identity_3ИвRead/ReadVariableOpвRead_1/ReadVariableOpвRead_2/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:         @R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:         @q
Read/ReadVariableOpReadVariableOpread_readvariableop_resource*
_output_shapes
:		А*
dtype0[
IdentityIdentityRead/ReadVariableOp:value:0*
T0*
_output_shapes
:		Аu
Read_1/ReadVariableOpReadVariableOpread_1_readvariableop_resource*
_output_shapes
:	@А*
dtype0_

Identity_1IdentityRead_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	@Аq
Read_2/ReadVariableOpReadVariableOpread_2_readvariableop_resource*
_output_shapes	
:А*
dtype0[

Identity_2IdentityRead_2/ReadVariableOp:value:0*
T0*
_output_shapes	
:А▒
PartitionedCallPartitionedCallinputszeros:output:0zeros_1:output:0Identity:output:0Identity_1:output:0Identity_2:output:0*
Tin

2*
Tout	
2*
_collective_manager_ids
 *f
_output_shapesT
R:         @:         @:         @:         @: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *(
f#R!
__inference_standard_lstm_24684i

Identity_3IdentityPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         @М
NoOpNoOp^Read/ReadVariableOp^Read_1/ReadVariableOp^Read_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  	: : : 2*
Read/ReadVariableOpRead/ReadVariableOp2.
Read_1/ReadVariableOpRead_1/ReadVariableOp2.
Read_2/ReadVariableOpRead_2/ReadVariableOp:\ X
4
_output_shapes"
 :                  	
 
_user_specified_nameinputs
╩	
є
@__inference_dense_layer_call_and_return_conditional_losses_21592

inputs1
matmul_readvariableop_resource:	@э.
biasadd_readvariableop_resource:	э
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@э*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         эs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:э*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         э`
IdentityIdentityBiasAdd:output:0^NoOp*
T0*(
_output_shapes
:         эw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         @: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
Б
▒
$__inference_lstm_layer_call_fn_23230

inputs
unknown:		А
	unknown_0:	@А
	unknown_1:	А
identityИвStatefulPartitionedCallс
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *H
fCRA
?__inference_lstm_layer_call_and_return_conditional_losses_21574o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         @`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  	: : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :                  	
 
_user_specified_nameinputs
╟@
╦
(__inference_gpu_lstm_with_fallback_23920

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4Иc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :                  	P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : o

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*+
_output_shapes
:         @R
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : s
ExpandDims_1
ExpandDimsinit_cExpandDims_1/dim:output:0*
T0*+
_output_shapes
:         @Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :И
splitSplitsplit/split_dim:output:0kernel*
T0*<
_output_shapes*
(:	@:	@:	@:	@*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ц
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*<
_output_shapes*
(:@@:@@:@@:@@*
	num_splitY

zeros_likeConst*
_output_shapes	
:А*
dtype0*
valueBА*    M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : r
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes	
:АS
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Э
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*D
_output_shapes2
0:@:@:@:@:@:@:@:@*
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
         a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes

:@	Y
ReshapeReshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:└a
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes

:@	[
	Reshape_1Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:└a
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

:@	[
	Reshape_2Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:└a
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes

:@	[
	Reshape_3Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:└a
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:@@[
	Reshape_4Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:А a
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0*
_output_shapes

:@@[
	Reshape_5Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:А a
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0*
_output_shapes

:@@[
	Reshape_6Reshapetranspose_7:y:0Const:output:0*
T0*
_output_shapes	
:А a
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0*
_output_shapes

:@@[
	Reshape_7Reshapetranspose_8:y:0Const:output:0*
T0*
_output_shapes	
:А [
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
:@[
	Reshape_9Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
:@\

Reshape_10Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
:@\

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
:@\

Reshape_12Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
:@\

Reshape_13Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
:@\

Reshape_14Reshapesplit_2:output:6Const:output:0*
T0*
_output_shapes
:@\

Reshape_15Reshapesplit_2:output:7Const:output:0*
T0*
_output_shapes
:@O
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : а
concat_1ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes

:АЦ╙
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1:output:0*
T0*f
_output_shapesT
R:                  @:         @:         @:f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
         _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:х
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         @*
shrink_axis_maskp
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:         @*
squeeze_dims
 r
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*'
_output_shapes
:         @*
squeeze_dims
 R
ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B :Г
ExpandDims_2
ExpandDimsstrided_slice:output:0ExpandDims_2/dim:output:0*
T0*+
_output_shapes
:         @[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @^
IdentityIdentitystrided_slice:output:0*
T0*'
_output_shapes
:         @c

Identity_1IdentityExpandDims_2:output:0*
T0*+
_output_shapes
:         @Z

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:         @\

Identity_3IdentitySqueeze_1:output:0*
T0*'
_output_shapes
:         @I

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:                  	:         @:         @:		А:	@А:А*=
api_implements+)lstm_5a52159d-dcb6-440d-9622-97ba434935b9*
api_preferred_deviceGPU*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :                  	
 
_user_specified_nameinputs:OK
'
_output_shapes
:         @
 
_user_specified_nameinit_h:OK
'
_output_shapes
:         @
 
_user_specified_nameinit_c:GC

_output_shapes
:		А
 
_user_specified_namekernel:QM

_output_shapes
:	@А
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:А

_user_specified_namebias
М	
╝
while_cond_20323
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice3
/while_while_cond_20323___redundant_placeholder03
/while_while_cond_20323___redundant_placeholder13
/while_while_cond_20323___redundant_placeholder23
/while_while_cond_20323___redundant_placeholder3
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :         @:         @: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         @:-)
'
_output_shapes
:         @:

_output_shapes
: :

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
╗;
┐
__inference_standard_lstm_21301

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :                  	B
ShapeShapetranspose:y:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ▓
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    	   р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:щ
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         	*
shrink_axis_maske
MatMulMatMulstrided_slice_1:output:0kernel*
T0*(
_output_shapes
:         А_
MatMul_1MatMulinit_hrecurrent_kernel*
T0*(
_output_shapes
:         Аe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         АT
BiasAddBiasAddadd:z:0bias*
T0*(
_output_shapes
:         АQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :╢
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:         @:         @:         @:         @*
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         @V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         @S
mulMulSigmoid_1:y:0init_c*
T0*'
_output_shapes
:         @N
TanhTanhsplit:output:2*
T0*'
_output_shapes
:         @U
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:         @T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         @V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         @K
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:         @Y
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:         @n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :┼
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ┤
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hinit_cstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelrecurrent_kernelbias*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*c
_output_shapesQ
O: : : : :         @:         @: : :		А:	@А:А* 
_read_only_resource_inputs
 *
_stateful_parallelism( *
bodyR
while_body_21215*
condR
while_cond_21214*b
output_shapesQ
O: : : : :         @:         @: : :		А:	@А:А*
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   ╓
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         @*
element_dtype0*
num_elementsh
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:З
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         @*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ц
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:         @[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  А?`
IdentityIdentitystrided_slice_2:output:0*
T0*'
_output_shapes
:         @]

Identity_1Identitytranspose_1:y:0*
T0*+
_output_shapes
:         @X

Identity_2Identitywhile:output:4*
T0*'
_output_shapes
:         @X

Identity_3Identitywhile:output:5*
T0*'
_output_shapes
:         @I

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:                  	:         @:         @:		А:	@А:А*=
api_implements+)lstm_e8470502-1520-4f1d-96c5-a71b9a8e2c85*
api_preferred_deviceCPU*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :                  	
 
_user_specified_nameinputs:OK
'
_output_shapes
:         @
 
_user_specified_nameinit_h:OK
'
_output_shapes
:         @
 
_user_specified_nameinit_c:GC

_output_shapes
:		А
 
_user_specified_namekernel:QM

_output_shapes
:	@А
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:А

_user_specified_namebias
ёK
а
&__forward_gpu_lstm_with_fallback_24525

inputs
init_h_0
init_c_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4
strided_slice
cudnnrnn

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
expanddims_1
concat_1
transpose_perm

init_h

init_c
concat_1_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
transpose_7_perm
transpose_8_perm
split_2_split_dim
split_split_dim
split_1_split_dim
concat_axisИc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          B
transpose_0	Transposeinputstranspose/perm:output:0*
T0P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : q

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0*+
_output_shapes
:         @R
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : u
ExpandDims_1
ExpandDimsinit_c_0ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:         @Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :И
splitSplitsplit/split_dim:output:0kernel*
T0*<
_output_shapes*
(:	@:	@:	@:	@*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ц
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*<
_output_shapes*
(:@@:@@:@@:@@*
	num_splitY

zeros_likeConst*
_output_shapes	
:А*
dtype0*
valueBА*    M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : r
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes	
:АS
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Э
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*D
_output_shapes2
0:@:@:@:@:@:@:@:@*
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
         a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes

:@	Y
ReshapeReshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:└a
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes

:@	[
	Reshape_1Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:└a
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

:@	[
	Reshape_2Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:└a
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes

:@	[
	Reshape_3Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:└a
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:@@[
	Reshape_4Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:А a
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0*
_output_shapes

:@@[
	Reshape_5Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:А a
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0*
_output_shapes

:@@[
	Reshape_6Reshapetranspose_7:y:0Const:output:0*
T0*
_output_shapes	
:А a
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0*
_output_shapes

:@@[
	Reshape_7Reshapetranspose_8:y:0Const:output:0*
T0*
_output_shapes	
:А [
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
:@[
	Reshape_9Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
:@\

Reshape_10Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
:@\

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
:@\

Reshape_12Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
:@\

Reshape_13Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
:@\

Reshape_14Reshapesplit_2:output:6Const:output:0*
T0*
_output_shapes
:@\

Reshape_15Reshapesplit_2:output:7Const:output:0*
T0*
_output_shapes
:@O
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Д

concat_1_0ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0╫
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1_0:output:0*
T0*f
_output_shapesT
R:                  @:         @:         @:f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
         _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╛
strided_slice_0StridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_maskp
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:         @*
squeeze_dims
 r
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*'
_output_shapes
:         @*
squeeze_dims
 R
ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B :Е
ExpandDims_2
ExpandDimsstrided_slice_0:output:0ExpandDims_2/dim:output:0*
T0*+
_output_shapes
:         @[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @`
IdentityIdentitystrided_slice_0:output:0*
T0*'
_output_shapes
:         @c

Identity_1IdentityExpandDims_2:output:0*
T0*+
_output_shapes
:         @Z

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:         @\

Identity_3IdentitySqueeze_1:output:0*
T0*'
_output_shapes
:         @I

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
concat_1concat_1_0:output:0"'
concat_1_axisconcat_1/axis:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output_h:0"!

cudnnrnn_0CudnnRNN:output_c:0"

cudnnrnn_1CudnnRNN:output:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"!

expanddimsExpandDims:output:0"%
expanddims_1ExpandDims_1:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
init_cinit_c_0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0")
strided_slicestrided_slice_0:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0"-
transpose_7_permtranspose_7/perm:output:0"-
transpose_8_permtranspose_8/perm:output:0")
transpose_permtranspose/perm:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:                  	:         @:         @:		А:	@А:А*=
api_implements+)lstm_e5590ddf-237b-46f1-9660-fe95f9321bd6*
api_preferred_deviceGPU*U
backward_function_name;9__inference___backward_gpu_lstm_with_fallback_24350_24526*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :                  	
 
_user_specified_nameinputs:OK
'
_output_shapes
:         @
 
_user_specified_nameinit_h:OK
'
_output_shapes
:         @
 
_user_specified_nameinit_c:GC

_output_shapes
:		А
 
_user_specified_namekernel:QM

_output_shapes
:	@А
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:А

_user_specified_namebias
ёK
а
&__forward_gpu_lstm_with_fallback_23667

inputs
init_h_0
init_c_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4
strided_slice
cudnnrnn

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
expanddims_1
concat_1
transpose_perm

init_h

init_c
concat_1_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
transpose_7_perm
transpose_8_perm
split_2_split_dim
split_split_dim
split_1_split_dim
concat_axisИc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          B
transpose_0	Transposeinputstranspose/perm:output:0*
T0P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : q

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0*+
_output_shapes
:         @R
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : u
ExpandDims_1
ExpandDimsinit_c_0ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:         @Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :И
splitSplitsplit/split_dim:output:0kernel*
T0*<
_output_shapes*
(:	@:	@:	@:	@*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ц
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*<
_output_shapes*
(:@@:@@:@@:@@*
	num_splitY

zeros_likeConst*
_output_shapes	
:А*
dtype0*
valueBА*    M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : r
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes	
:АS
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Э
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*D
_output_shapes2
0:@:@:@:@:@:@:@:@*
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
         a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes

:@	Y
ReshapeReshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:└a
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes

:@	[
	Reshape_1Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:└a
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

:@	[
	Reshape_2Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:└a
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes

:@	[
	Reshape_3Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:└a
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:@@[
	Reshape_4Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:А a
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0*
_output_shapes

:@@[
	Reshape_5Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:А a
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0*
_output_shapes

:@@[
	Reshape_6Reshapetranspose_7:y:0Const:output:0*
T0*
_output_shapes	
:А a
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0*
_output_shapes

:@@[
	Reshape_7Reshapetranspose_8:y:0Const:output:0*
T0*
_output_shapes	
:А [
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
:@[
	Reshape_9Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
:@\

Reshape_10Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
:@\

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
:@\

Reshape_12Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
:@\

Reshape_13Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
:@\

Reshape_14Reshapesplit_2:output:6Const:output:0*
T0*
_output_shapes
:@\

Reshape_15Reshapesplit_2:output:7Const:output:0*
T0*
_output_shapes
:@O
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Д

concat_1_0ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0╫
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1_0:output:0*
T0*f
_output_shapesT
R:                  @:         @:         @:f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
         _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╛
strided_slice_0StridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_maskp
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:         @*
squeeze_dims
 r
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*'
_output_shapes
:         @*
squeeze_dims
 R
ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B :Е
ExpandDims_2
ExpandDimsstrided_slice_0:output:0ExpandDims_2/dim:output:0*
T0*+
_output_shapes
:         @[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @`
IdentityIdentitystrided_slice_0:output:0*
T0*'
_output_shapes
:         @c

Identity_1IdentityExpandDims_2:output:0*
T0*+
_output_shapes
:         @Z

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:         @\

Identity_3IdentitySqueeze_1:output:0*
T0*'
_output_shapes
:         @I

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
concat_1concat_1_0:output:0"'
concat_1_axisconcat_1/axis:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output_h:0"!

cudnnrnn_0CudnnRNN:output_c:0"

cudnnrnn_1CudnnRNN:output:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"!

expanddimsExpandDims:output:0"%
expanddims_1ExpandDims_1:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
init_cinit_c_0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0")
strided_slicestrided_slice_0:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0"-
transpose_7_permtranspose_7/perm:output:0"-
transpose_8_permtranspose_8/perm:output:0")
transpose_permtranspose/perm:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:                  	:         @:         @:		А:	@А:А*=
api_implements+)lstm_d60d0f35-f731-4def-8ea6-8cb0eeff7cf2*
api_preferred_deviceGPU*U
backward_function_name;9__inference___backward_gpu_lstm_with_fallback_23492_23668*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :                  	
 
_user_specified_nameinputs:OK
'
_output_shapes
:         @
 
_user_specified_nameinit_h:OK
'
_output_shapes
:         @
 
_user_specified_nameinit_c:GC

_output_shapes
:		А
 
_user_specified_namekernel:QM

_output_shapes
:	@А
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:А

_user_specified_namebias
Г┬
у
9__inference___backward_gpu_lstm_with_fallback_20945_21121
placeholder
placeholder_1
placeholder_2
placeholder_3
placeholder_43
/gradients_expanddims_2_grad_shape_strided_slice)
%gradients_squeeze_grad_shape_cudnnrnn+
'gradients_squeeze_1_grad_shape_cudnnrnn/
+gradients_strided_slice_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9
5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_15
1gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h,
(gradients_expanddims_1_grad_shape_init_c-
)gradients_concat_1_grad_mod_concat_1_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_permA
=gradients_transpose_7_grad_invertpermutation_transpose_7_permA
=gradients_transpose_8_grad_invertpermutation_transpose_8_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim)
%gradients_concat_grad_mod_concat_axis
identity

identity_1

identity_2

identity_3

identity_4

identity_5И^
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:         @d
gradients/grad_ys_1Identityplaceholder_1*
T0*+
_output_shapes
:         @`
gradients/grad_ys_2Identityplaceholder_2*
T0*'
_output_shapes
:         @`
gradients/grad_ys_3Identityplaceholder_3*
T0*'
_output_shapes
:         @O
gradients/grad_ys_4Identityplaceholder_4*
T0*
_output_shapes
: А
!gradients/ExpandDims_2_grad/ShapeShape/gradients_expanddims_2_grad_shape_strided_slice*
T0*
_output_shapes
:к
#gradients/ExpandDims_2_grad/ReshapeReshapegradients/grad_ys_1:output:0*gradients/ExpandDims_2_grad/Shape:output:0*
T0*'
_output_shapes
:         @q
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
:д
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*+
_output_shapes
:         @u
gradients/Squeeze_1_grad/ShapeShape'gradients_squeeze_1_grad_shape_cudnnrnn*
T0*
_output_shapes
:и
 gradients/Squeeze_1_grad/ReshapeReshapegradients/grad_ys_3:output:0'gradients/Squeeze_1_grad/Shape:output:0*
T0*+
_output_shapes
:         @┼
gradients/AddNAddNgradients/grad_ys_0:output:0,gradients/ExpandDims_2_grad/Reshape:output:0*
N*
T0*&
_class
loc:@gradients/grad_ys_0*'
_output_shapes
:         @}
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:Ж
3gradients/strided_slice_grad/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
         {
1gradients/strided_slice_grad/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 
5gradients/strided_slice_grad/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:в
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0<gradients/strided_slice_grad/StridedSliceGrad/begin:output:0:gradients/strided_slice_grad/StridedSliceGrad/end:output:0>gradients/strided_slice_grad/StridedSliceGrad/strides:output:0gradients/AddN:sum:0*
Index0*
T0*4
_output_shapes"
 :                  @*
shrink_axis_maskc
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
:м
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_11gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnn'gradients_squeeze_1_grad_shape_cudnnrnn6gradients/strided_slice_grad/StridedSliceGrad:output:0'gradients/Squeeze_grad/Reshape:output:0)gradients/Squeeze_1_grad/Reshape:output:0gradients_zeros_like_cudnnrnn*
T0*j
_output_shapesX
V:                  	:         @:         @:АЦЦ
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:┘
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*4
_output_shapes"
 :                  	u
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:┼
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*'
_output_shapes
:         @y
!gradients/ExpandDims_1_grad/ShapeShape(gradients_expanddims_1_grad_shape_init_c*
T0*
_output_shapes
:╔
#gradients/ExpandDims_1_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_c_backprop:0*gradients/ExpandDims_1_grad/Shape:output:0*
T0*'
_output_shapes
:         @^
gradients/concat_1_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :Ъ
gradients/concat_1_grad/modFloorMod)gradients_concat_1_grad_mod_concat_1_axis%gradients/concat_1_grad/Rank:output:0*
T0*
_output_shapes
: h
gradients/concat_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:└j
gradients/concat_1_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:└j
gradients/concat_1_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:└j
gradients/concat_1_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:└j
gradients/concat_1_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:А j
gradients/concat_1_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:А j
gradients/concat_1_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB:А j
gradients/concat_1_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB:А i
gradients/concat_1_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB:@i
gradients/concat_1_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB:@j
 gradients/concat_1_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB:@j
 gradients/concat_1_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB:@j
 gradients/concat_1_grad/Shape_12Const*
_output_shapes
:*
dtype0*
valueB:@j
 gradients/concat_1_grad/Shape_13Const*
_output_shapes
:*
dtype0*
valueB:@j
 gradients/concat_1_grad/Shape_14Const*
_output_shapes
:*
dtype0*
valueB:@j
 gradients/concat_1_grad/Shape_15Const*
_output_shapes
:*
dtype0*
valueB:@°
$gradients/concat_1_grad/ConcatOffsetConcatOffsetgradients/concat_1_grad/mod:z:0&gradients/concat_1_grad/Shape:output:0(gradients/concat_1_grad/Shape_1:output:0(gradients/concat_1_grad/Shape_2:output:0(gradients/concat_1_grad/Shape_3:output:0(gradients/concat_1_grad/Shape_4:output:0(gradients/concat_1_grad/Shape_5:output:0(gradients/concat_1_grad/Shape_6:output:0(gradients/concat_1_grad/Shape_7:output:0(gradients/concat_1_grad/Shape_8:output:0(gradients/concat_1_grad/Shape_9:output:0)gradients/concat_1_grad/Shape_10:output:0)gradients/concat_1_grad/Shape_11:output:0)gradients/concat_1_grad/Shape_12:output:0)gradients/concat_1_grad/Shape_13:output:0)gradients/concat_1_grad/Shape_14:output:0)gradients/concat_1_grad/Shape_15:output:0*
N*t
_output_shapesb
`::::::::::::::::ь
gradients/concat_1_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:0&gradients/concat_1_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:└Ё
gradients/concat_1_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:1(gradients/concat_1_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:└Ё
gradients/concat_1_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:2(gradients/concat_1_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:└Ё
gradients/concat_1_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:3(gradients/concat_1_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:└Ё
gradients/concat_1_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:4(gradients/concat_1_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:А Ё
gradients/concat_1_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:5(gradients/concat_1_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:А Ё
gradients/concat_1_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:6(gradients/concat_1_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes	
:А Ё
gradients/concat_1_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:7(gradients/concat_1_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes	
:А я
gradients/concat_1_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:8(gradients/concat_1_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
:@я
gradients/concat_1_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:9(gradients/concat_1_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
:@Є
 gradients/concat_1_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:10)gradients/concat_1_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
:@Є
 gradients/concat_1_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:11)gradients/concat_1_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
:@Є
 gradients/concat_1_grad/Slice_12Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:12)gradients/concat_1_grad/Shape_12:output:0*
Index0*
T0*
_output_shapes
:@Є
 gradients/concat_1_grad/Slice_13Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:13)gradients/concat_1_grad/Shape_13:output:0*
Index0*
T0*
_output_shapes
:@Є
 gradients/concat_1_grad/Slice_14Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:14)gradients/concat_1_grad/Shape_14:output:0*
Index0*
T0*
_output_shapes
:@Є
 gradients/concat_1_grad/Slice_15Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:15)gradients/concat_1_grad/Shape_15:output:0*
Index0*
T0*
_output_shapes
:@m
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"@   	   б
gradients/Reshape_grad/ReshapeReshape&gradients/concat_1_grad/Slice:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes

:@	o
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"@   	   з
 gradients/Reshape_1_grad/ReshapeReshape(gradients/concat_1_grad/Slice_1:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes

:@	o
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"@   	   з
 gradients/Reshape_2_grad/ReshapeReshape(gradients/concat_1_grad/Slice_2:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes

:@	o
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"@   	   з
 gradients/Reshape_3_grad/ReshapeReshape(gradients/concat_1_grad/Slice_3:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes

:@	o
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"@   @   з
 gradients/Reshape_4_grad/ReshapeReshape(gradients/concat_1_grad/Slice_4:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:@@o
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"@   @   з
 gradients/Reshape_5_grad/ReshapeReshape(gradients/concat_1_grad/Slice_5:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:@@o
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"@   @   з
 gradients/Reshape_6_grad/ReshapeReshape(gradients/concat_1_grad/Slice_6:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:@@o
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"@   @   з
 gradients/Reshape_7_grad/ReshapeReshape(gradients/concat_1_grad/Slice_7:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes

:@@h
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:@г
 gradients/Reshape_8_grad/ReshapeReshape(gradients/concat_1_grad/Slice_8:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
:@h
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:@г
 gradients/Reshape_9_grad/ReshapeReshape(gradients/concat_1_grad/Slice_9:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
:@i
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:@ж
!gradients/Reshape_10_grad/ReshapeReshape)gradients/concat_1_grad/Slice_10:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
:@i
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:@ж
!gradients/Reshape_11_grad/ReshapeReshape)gradients/concat_1_grad/Slice_11:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
:@i
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:@ж
!gradients/Reshape_12_grad/ReshapeReshape)gradients/concat_1_grad/Slice_12:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
:@i
gradients/Reshape_13_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:@ж
!gradients/Reshape_13_grad/ReshapeReshape)gradients/concat_1_grad/Slice_13:output:0(gradients/Reshape_13_grad/Shape:output:0*
T0*
_output_shapes
:@i
gradients/Reshape_14_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:@ж
!gradients/Reshape_14_grad/ReshapeReshape)gradients/concat_1_grad/Slice_14:output:0(gradients/Reshape_14_grad/Shape:output:0*
T0*
_output_shapes
:@i
gradients/Reshape_15_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:@ж
!gradients/Reshape_15_grad/ReshapeReshape)gradients/concat_1_grad/Slice_15:output:0(gradients/Reshape_15_grad/Shape:output:0*
T0*
_output_shapes
:@Ь
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:╡
$gradients/transpose_1_grad/transpose	Transpose'gradients/Reshape_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes

:	@Ь
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:╖
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes

:	@Ь
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:╖
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes

:	@Ь
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:╖
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes

:	@Ь
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:╖
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@@Ь
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:╖
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@@Ь
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:╖
$gradients/transpose_7_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@@Ь
,gradients/transpose_8_grad/InvertPermutationInvertPermutation=gradients_transpose_8_grad_invertpermutation_transpose_8_perm*
_output_shapes
:╖
$gradients/transpose_8_grad/transpose	Transpose)gradients/Reshape_7_grad/Reshape:output:00gradients/transpose_8_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@@ч
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0*gradients/Reshape_13_grad/Reshape:output:0*gradients/Reshape_14_grad/Reshape:output:0*gradients/Reshape_15_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:Ап
gradients/split_grad/concatConcatV2(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0*
_output_shapes
:		А╡
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0(gradients/transpose_7_grad/transpose:y:0(gradients/transpose_8_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0*
_output_shapes
:	@А\
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :Т
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: f
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:Аh
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:А╩
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0*
N* 
_output_shapes
::╥
gradients/concat_grad/SliceSlice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:А╓
gradients/concat_grad/Slice_1Slice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:А{
IdentityIdentity&gradients/transpose_grad/transpose:y:0*
T0*4
_output_shapes"
 :                  	t

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0*
T0*'
_output_shapes
:         @v

Identity_2Identity,gradients/ExpandDims_1_grad/Reshape:output:0*
T0*'
_output_shapes
:         @f

Identity_3Identity$gradients/split_grad/concat:output:0*
T0*
_output_shapes
:		Аh

Identity_4Identity&gradients/split_1_grad/concat:output:0*
T0*
_output_shapes
:	@Аd

Identity_5Identity&gradients/concat_grad/Slice_1:output:0*
T0*
_output_shapes	
:А"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*(
_construction_contextkEagerRuntime*И
_input_shapesЎ
є:         @:         @:         @:         @: :         @:         @:         @:                  @::                  	:         @:         @:АЦ::         @:         @: ::::::::: : : : *=
api_implements+)lstm_733a4ab0-824c-43e0-83a3-926a8e9369c1*
api_preferred_deviceGPU*A
forward_function_name(&__forward_gpu_lstm_with_fallback_21120*
go_backwards( *

time_major( :- )
'
_output_shapes
:         @:1-
+
_output_shapes
:         @:-)
'
_output_shapes
:         @:-)
'
_output_shapes
:         @:

_output_shapes
: :-)
'
_output_shapes
:         @:1-
+
_output_shapes
:         @:1-
+
_output_shapes
:         @::6
4
_output_shapes"
 :                  @:	

_output_shapes
:::
6
4
_output_shapes"
 :                  	:1-
+
_output_shapes
:         @:1-
+
_output_shapes
:         @:"

_output_shapes

:АЦ: 

_output_shapes
::-)
'
_output_shapes
:         @:-)
'
_output_shapes
:         @:

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
М	
╝
while_cond_24168
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice3
/while_while_cond_24168___redundant_placeholder03
/while_while_cond_24168___redundant_placeholder13
/while_while_cond_24168___redundant_placeholder23
/while_while_cond_24168___redundant_placeholder3
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :         @:         @: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         @:-)
'
_output_shapes
:         @:

_output_shapes
: :

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
ёK
а
&__forward_gpu_lstm_with_fallback_22746

inputs
init_h_0
init_c_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4
strided_slice
cudnnrnn

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
expanddims_1
concat_1
transpose_perm

init_h

init_c
concat_1_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
transpose_7_perm
transpose_8_perm
split_2_split_dim
split_split_dim
split_1_split_dim
concat_axisИc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          B
transpose_0	Transposeinputstranspose/perm:output:0*
T0P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : q

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0*+
_output_shapes
:         @R
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : u
ExpandDims_1
ExpandDimsinit_c_0ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:         @Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :И
splitSplitsplit/split_dim:output:0kernel*
T0*<
_output_shapes*
(:	@:	@:	@:	@*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ц
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*<
_output_shapes*
(:@@:@@:@@:@@*
	num_splitY

zeros_likeConst*
_output_shapes	
:А*
dtype0*
valueBА*    M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : r
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes	
:АS
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Э
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*D
_output_shapes2
0:@:@:@:@:@:@:@:@*
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
         a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes

:@	Y
ReshapeReshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:└a
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes

:@	[
	Reshape_1Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:└a
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

:@	[
	Reshape_2Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:└a
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes

:@	[
	Reshape_3Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:└a
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:@@[
	Reshape_4Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:А a
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0*
_output_shapes

:@@[
	Reshape_5Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:А a
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0*
_output_shapes

:@@[
	Reshape_6Reshapetranspose_7:y:0Const:output:0*
T0*
_output_shapes	
:А a
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0*
_output_shapes

:@@[
	Reshape_7Reshapetranspose_8:y:0Const:output:0*
T0*
_output_shapes	
:А [
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
:@[
	Reshape_9Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
:@\

Reshape_10Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
:@\

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
:@\

Reshape_12Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
:@\

Reshape_13Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
:@\

Reshape_14Reshapesplit_2:output:6Const:output:0*
T0*
_output_shapes
:@\

Reshape_15Reshapesplit_2:output:7Const:output:0*
T0*
_output_shapes
:@O
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Д

concat_1_0ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0╫
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1_0:output:0*
T0*f
_output_shapesT
R:                  @:         @:         @:f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
         _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╛
strided_slice_0StridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_maskp
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:         @*
squeeze_dims
 r
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*'
_output_shapes
:         @*
squeeze_dims
 R
ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B :Е
ExpandDims_2
ExpandDimsstrided_slice_0:output:0ExpandDims_2/dim:output:0*
T0*+
_output_shapes
:         @[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @`
IdentityIdentitystrided_slice_0:output:0*
T0*'
_output_shapes
:         @c

Identity_1IdentityExpandDims_2:output:0*
T0*+
_output_shapes
:         @Z

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:         @\

Identity_3IdentitySqueeze_1:output:0*
T0*'
_output_shapes
:         @I

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
concat_1concat_1_0:output:0"'
concat_1_axisconcat_1/axis:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output_h:0"!

cudnnrnn_0CudnnRNN:output_c:0"

cudnnrnn_1CudnnRNN:output:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"!

expanddimsExpandDims:output:0"%
expanddims_1ExpandDims_1:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
init_cinit_c_0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0")
strided_slicestrided_slice_0:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0"-
transpose_7_permtranspose_7/perm:output:0"-
transpose_8_permtranspose_8/perm:output:0")
transpose_permtranspose/perm:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:                  	:         @:         @:		А:	@А:А*=
api_implements+)lstm_b4638b8b-6e40-4922-ab41-0ce5301502aa*
api_preferred_deviceGPU*U
backward_function_name;9__inference___backward_gpu_lstm_with_fallback_22571_22747*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :                  	
 
_user_specified_nameinputs:OK
'
_output_shapes
:         @
 
_user_specified_nameinit_h:OK
'
_output_shapes
:         @
 
_user_specified_nameinit_c:GC

_output_shapes
:		А
 
_user_specified_namekernel:QM

_output_shapes
:	@А
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:А

_user_specified_namebias
╟@
╦
(__inference_gpu_lstm_with_fallback_24778

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4Иc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :                  	P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : o

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*+
_output_shapes
:         @R
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : s
ExpandDims_1
ExpandDimsinit_cExpandDims_1/dim:output:0*
T0*+
_output_shapes
:         @Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :И
splitSplitsplit/split_dim:output:0kernel*
T0*<
_output_shapes*
(:	@:	@:	@:	@*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ц
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*<
_output_shapes*
(:@@:@@:@@:@@*
	num_splitY

zeros_likeConst*
_output_shapes	
:А*
dtype0*
valueBА*    M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : r
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes	
:АS
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Э
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*D
_output_shapes2
0:@:@:@:@:@:@:@:@*
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
         a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes

:@	Y
ReshapeReshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:└a
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes

:@	[
	Reshape_1Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:└a
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

:@	[
	Reshape_2Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:└a
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes

:@	[
	Reshape_3Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:└a
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:@@[
	Reshape_4Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:А a
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0*
_output_shapes

:@@[
	Reshape_5Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:А a
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0*
_output_shapes

:@@[
	Reshape_6Reshapetranspose_7:y:0Const:output:0*
T0*
_output_shapes	
:А a
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0*
_output_shapes

:@@[
	Reshape_7Reshapetranspose_8:y:0Const:output:0*
T0*
_output_shapes	
:А [
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
:@[
	Reshape_9Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
:@\

Reshape_10Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
:@\

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
:@\

Reshape_12Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
:@\

Reshape_13Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
:@\

Reshape_14Reshapesplit_2:output:6Const:output:0*
T0*
_output_shapes
:@\

Reshape_15Reshapesplit_2:output:7Const:output:0*
T0*
_output_shapes
:@O
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : а
concat_1ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes

:АЦ╙
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1:output:0*
T0*f
_output_shapesT
R:                  @:         @:         @:f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
         _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:х
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         @*
shrink_axis_maskp
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:         @*
squeeze_dims
 r
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*'
_output_shapes
:         @*
squeeze_dims
 R
ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B :Г
ExpandDims_2
ExpandDimsstrided_slice:output:0ExpandDims_2/dim:output:0*
T0*+
_output_shapes
:         @[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @^
IdentityIdentitystrided_slice:output:0*
T0*'
_output_shapes
:         @c

Identity_1IdentityExpandDims_2:output:0*
T0*+
_output_shapes
:         @Z

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:         @\

Identity_3IdentitySqueeze_1:output:0*
T0*'
_output_shapes
:         @I

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:                  	:         @:         @:		А:	@А:А*=
api_implements+)lstm_8dc90f36-b787-4459-a772-dec1d233e41f*
api_preferred_deviceGPU*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :                  	
 
_user_specified_nameinputs:OK
'
_output_shapes
:         @
 
_user_specified_nameinit_h:OK
'
_output_shapes
:         @
 
_user_specified_nameinit_c:GC

_output_shapes
:		А
 
_user_specified_namekernel:QM

_output_shapes
:	@А
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:А

_user_specified_namebias
к
╗
?__inference_lstm_layer_call_and_return_conditional_losses_24099
inputs_0/
read_readvariableop_resource:		А1
read_1_readvariableop_resource:	@А-
read_2_readvariableop_resource:	А

identity_3ИвRead/ReadVariableOpвRead_1/ReadVariableOpвRead_2/ReadVariableOp=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:         @R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:         @q
Read/ReadVariableOpReadVariableOpread_readvariableop_resource*
_output_shapes
:		А*
dtype0[
IdentityIdentityRead/ReadVariableOp:value:0*
T0*
_output_shapes
:		Аu
Read_1/ReadVariableOpReadVariableOpread_1_readvariableop_resource*
_output_shapes
:	@А*
dtype0_

Identity_1IdentityRead_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	@Аq
Read_2/ReadVariableOpReadVariableOpread_2_readvariableop_resource*
_output_shapes	
:А*
dtype0[

Identity_2IdentityRead_2/ReadVariableOp:value:0*
T0*
_output_shapes	
:А│
PartitionedCallPartitionedCallinputs_0zeros:output:0zeros_1:output:0Identity:output:0Identity_1:output:0Identity_2:output:0*
Tin

2*
Tout	
2*
_collective_manager_ids
 *f
_output_shapesT
R:         @:         @:         @:         @: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *(
f#R!
__inference_standard_lstm_23826i

Identity_3IdentityPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         @М
NoOpNoOp^Read/ReadVariableOp^Read_1/ReadVariableOp^Read_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  	: : : 2*
Read/ReadVariableOpRead/ReadVariableOp2.
Read_1/ReadVariableOpRead_1/ReadVariableOp2.
Read_2/ReadVariableOpRead_2/ReadVariableOp:^ Z
4
_output_shapes"
 :                  	
"
_user_specified_name
inputs/0
╪)
╬
while_body_24169
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_bias_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel#
while_matmul_1_recurrent_kernel
while_biasadd_biasИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    	   ж
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         	*
element_dtype0Т
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*(
_output_shapes
:         АГ
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*(
_output_shapes
:         Аw
	while/addAddV2while/MatMul:product:0while/MatMul_1:product:0*
T0*(
_output_shapes
:         Аp
while/BiasAddBiasAddwhile/add:z:0while_biasadd_bias_0*
T0*(
_output_shapes
:         АW
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :╚
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*`
_output_shapesN
L:         @:         @:         @:         @*
	num_split`
while/SigmoidSigmoidwhile/split:output:0*
T0*'
_output_shapes
:         @b
while/Sigmoid_1Sigmoidwhile/split:output:1*
T0*'
_output_shapes
:         @l
	while/mulMulwhile/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:         @Z

while/TanhTanhwhile/split:output:2*
T0*'
_output_shapes
:         @g
while/mul_1Mulwhile/Sigmoid:y:0while/Tanh:y:0*
T0*'
_output_shapes
:         @f
while/add_1AddV2while/mul:z:0while/mul_1:z:0*
T0*'
_output_shapes
:         @b
while/Sigmoid_2Sigmoidwhile/split:output:3*
T0*'
_output_shapes
:         @W
while/Tanh_1Tanhwhile/add_1:z:0*
T0*'
_output_shapes
:         @k
while/mul_2Mulwhile/Sigmoid_2:y:0while/Tanh_1:y:0*
T0*'
_output_shapes
:         @r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : р
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/mul_2:z:0*
_output_shapes
: *
element_dtype0:щш╥O
while/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :`
while/add_2AddV2while_placeholderwhile/add_2/y:output:0*
T0*
_output_shapes
: O
while/add_3/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_3AddV2while_while_loop_counterwhile/add_3/y:output:0*
T0*
_output_shapes
: L
while/IdentityIdentitywhile/add_3:z:0*
T0*
_output_shapes
: ]
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: N
while/Identity_2Identitywhile/add_2:z:0*
T0*
_output_shapes
: y
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: _
while/Identity_4Identitywhile/mul_2:z:0*
T0*'
_output_shapes
:         @_
while/Identity_5Identitywhile/add_1:z:0*
T0*'
_output_shapes
:         @"*
while_biasadd_biaswhile_biasadd_bias_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O: : : : :         @:         @: : :		А:	@А:А: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         @:-)
'
_output_shapes
:         @:

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:		А:%	!

_output_shapes
:	@А:!


_output_shapes	
:А
╗;
┐
__inference_standard_lstm_23397

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :                  	B
ShapeShapetranspose:y:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ▓
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    	   р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:щ
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         	*
shrink_axis_maske
MatMulMatMulstrided_slice_1:output:0kernel*
T0*(
_output_shapes
:         А_
MatMul_1MatMulinit_hrecurrent_kernel*
T0*(
_output_shapes
:         Аe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         АT
BiasAddBiasAddadd:z:0bias*
T0*(
_output_shapes
:         АQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :╢
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:         @:         @:         @:         @*
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         @V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         @S
mulMulSigmoid_1:y:0init_c*
T0*'
_output_shapes
:         @N
TanhTanhsplit:output:2*
T0*'
_output_shapes
:         @U
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:         @T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         @V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         @K
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:         @Y
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:         @n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :┼
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ┤
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hinit_cstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelrecurrent_kernelbias*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*c
_output_shapesQ
O: : : : :         @:         @: : :		А:	@А:А* 
_read_only_resource_inputs
 *
_stateful_parallelism( *
bodyR
while_body_23311*
condR
while_cond_23310*b
output_shapesQ
O: : : : :         @:         @: : :		А:	@А:А*
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   ╓
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         @*
element_dtype0*
num_elementsh
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:З
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         @*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ц
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:         @[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  А?`
IdentityIdentitystrided_slice_2:output:0*
T0*'
_output_shapes
:         @]

Identity_1Identitytranspose_1:y:0*
T0*+
_output_shapes
:         @X

Identity_2Identitywhile:output:4*
T0*'
_output_shapes
:         @X

Identity_3Identitywhile:output:5*
T0*'
_output_shapes
:         @I

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:                  	:         @:         @:		А:	@А:А*=
api_implements+)lstm_d60d0f35-f731-4def-8ea6-8cb0eeff7cf2*
api_preferred_deviceCPU*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :                  	
 
_user_specified_nameinputs:OK
'
_output_shapes
:         @
 
_user_specified_nameinit_h:OK
'
_output_shapes
:         @
 
_user_specified_nameinit_c:GC

_output_shapes
:		А
 
_user_specified_namekernel:QM

_output_shapes
:	@А
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:А

_user_specified_namebias
╟@
╦
(__inference_gpu_lstm_with_fallback_20504

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4Иc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :                  	P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : o

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*+
_output_shapes
:         @R
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : s
ExpandDims_1
ExpandDimsinit_cExpandDims_1/dim:output:0*
T0*+
_output_shapes
:         @Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :И
splitSplitsplit/split_dim:output:0kernel*
T0*<
_output_shapes*
(:	@:	@:	@:	@*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ц
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*<
_output_shapes*
(:@@:@@:@@:@@*
	num_splitY

zeros_likeConst*
_output_shapes	
:А*
dtype0*
valueBА*    M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : r
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes	
:АS
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Э
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*D
_output_shapes2
0:@:@:@:@:@:@:@:@*
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
         a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes

:@	Y
ReshapeReshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:└a
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes

:@	[
	Reshape_1Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:└a
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

:@	[
	Reshape_2Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:└a
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes

:@	[
	Reshape_3Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:└a
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:@@[
	Reshape_4Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:А a
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0*
_output_shapes

:@@[
	Reshape_5Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:А a
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0*
_output_shapes

:@@[
	Reshape_6Reshapetranspose_7:y:0Const:output:0*
T0*
_output_shapes	
:А a
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0*
_output_shapes

:@@[
	Reshape_7Reshapetranspose_8:y:0Const:output:0*
T0*
_output_shapes	
:А [
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
:@[
	Reshape_9Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
:@\

Reshape_10Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
:@\

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
:@\

Reshape_12Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
:@\

Reshape_13Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
:@\

Reshape_14Reshapesplit_2:output:6Const:output:0*
T0*
_output_shapes
:@\

Reshape_15Reshapesplit_2:output:7Const:output:0*
T0*
_output_shapes
:@O
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : а
concat_1ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes

:АЦ╙
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1:output:0*
T0*f
_output_shapesT
R:                  @:         @:         @:f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
         _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:х
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         @*
shrink_axis_maskp
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:         @*
squeeze_dims
 r
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*'
_output_shapes
:         @*
squeeze_dims
 R
ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B :Г
ExpandDims_2
ExpandDimsstrided_slice:output:0ExpandDims_2/dim:output:0*
T0*+
_output_shapes
:         @[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @^
IdentityIdentitystrided_slice:output:0*
T0*'
_output_shapes
:         @c

Identity_1IdentityExpandDims_2:output:0*
T0*+
_output_shapes
:         @Z

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:         @\

Identity_3IdentitySqueeze_1:output:0*
T0*'
_output_shapes
:         @I

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:                  	:         @:         @:		А:	@А:А*=
api_implements+)lstm_8d97cef0-5325-4590-9604-659863d02e30*
api_preferred_deviceGPU*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :                  	
 
_user_specified_nameinputs:OK
'
_output_shapes
:         @
 
_user_specified_nameinit_h:OK
'
_output_shapes
:         @
 
_user_specified_nameinit_c:GC

_output_shapes
:		А
 
_user_specified_namekernel:QM

_output_shapes
:	@А
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:А

_user_specified_namebias
╟@
╦
(__inference_gpu_lstm_with_fallback_21889

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4Иc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :                  	P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : o

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*+
_output_shapes
:         @R
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : s
ExpandDims_1
ExpandDimsinit_cExpandDims_1/dim:output:0*
T0*+
_output_shapes
:         @Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :И
splitSplitsplit/split_dim:output:0kernel*
T0*<
_output_shapes*
(:	@:	@:	@:	@*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ц
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*<
_output_shapes*
(:@@:@@:@@:@@*
	num_splitY

zeros_likeConst*
_output_shapes	
:А*
dtype0*
valueBА*    M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : r
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes	
:АS
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Э
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*D
_output_shapes2
0:@:@:@:@:@:@:@:@*
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
         a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes

:@	Y
ReshapeReshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:└a
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes

:@	[
	Reshape_1Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:└a
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

:@	[
	Reshape_2Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:└a
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes

:@	[
	Reshape_3Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:└a
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:@@[
	Reshape_4Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:А a
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0*
_output_shapes

:@@[
	Reshape_5Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:А a
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0*
_output_shapes

:@@[
	Reshape_6Reshapetranspose_7:y:0Const:output:0*
T0*
_output_shapes	
:А a
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0*
_output_shapes

:@@[
	Reshape_7Reshapetranspose_8:y:0Const:output:0*
T0*
_output_shapes	
:А [
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
:@[
	Reshape_9Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
:@\

Reshape_10Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
:@\

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
:@\

Reshape_12Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
:@\

Reshape_13Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
:@\

Reshape_14Reshapesplit_2:output:6Const:output:0*
T0*
_output_shapes
:@\

Reshape_15Reshapesplit_2:output:7Const:output:0*
T0*
_output_shapes
:@O
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : а
concat_1ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes

:АЦ╙
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1:output:0*
T0*f
_output_shapesT
R:                  @:         @:         @:f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
         _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:х
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         @*
shrink_axis_maskp
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:         @*
squeeze_dims
 r
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*'
_output_shapes
:         @*
squeeze_dims
 R
ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B :Г
ExpandDims_2
ExpandDimsstrided_slice:output:0ExpandDims_2/dim:output:0*
T0*+
_output_shapes
:         @[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @^
IdentityIdentitystrided_slice:output:0*
T0*'
_output_shapes
:         @c

Identity_1IdentityExpandDims_2:output:0*
T0*+
_output_shapes
:         @Z

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:         @\

Identity_3IdentitySqueeze_1:output:0*
T0*'
_output_shapes
:         @I

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:                  	:         @:         @:		А:	@А:А*=
api_implements+)lstm_d4300e95-cf4b-4bbb-ba38-7f8ecfb78490*
api_preferred_deviceGPU*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :                  	
 
_user_specified_nameinputs:OK
'
_output_shapes
:         @
 
_user_specified_nameinit_h:OK
'
_output_shapes
:         @
 
_user_specified_nameinit_c:GC

_output_shapes
:		А
 
_user_specified_namekernel:QM

_output_shapes
:	@А
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:А

_user_specified_namebias
Г┬
у
9__inference___backward_gpu_lstm_with_fallback_24350_24526
placeholder
placeholder_1
placeholder_2
placeholder_3
placeholder_43
/gradients_expanddims_2_grad_shape_strided_slice)
%gradients_squeeze_grad_shape_cudnnrnn+
'gradients_squeeze_1_grad_shape_cudnnrnn/
+gradients_strided_slice_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9
5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_15
1gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h,
(gradients_expanddims_1_grad_shape_init_c-
)gradients_concat_1_grad_mod_concat_1_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_permA
=gradients_transpose_7_grad_invertpermutation_transpose_7_permA
=gradients_transpose_8_grad_invertpermutation_transpose_8_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim)
%gradients_concat_grad_mod_concat_axis
identity

identity_1

identity_2

identity_3

identity_4

identity_5И^
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:         @d
gradients/grad_ys_1Identityplaceholder_1*
T0*+
_output_shapes
:         @`
gradients/grad_ys_2Identityplaceholder_2*
T0*'
_output_shapes
:         @`
gradients/grad_ys_3Identityplaceholder_3*
T0*'
_output_shapes
:         @O
gradients/grad_ys_4Identityplaceholder_4*
T0*
_output_shapes
: А
!gradients/ExpandDims_2_grad/ShapeShape/gradients_expanddims_2_grad_shape_strided_slice*
T0*
_output_shapes
:к
#gradients/ExpandDims_2_grad/ReshapeReshapegradients/grad_ys_1:output:0*gradients/ExpandDims_2_grad/Shape:output:0*
T0*'
_output_shapes
:         @q
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
:д
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*+
_output_shapes
:         @u
gradients/Squeeze_1_grad/ShapeShape'gradients_squeeze_1_grad_shape_cudnnrnn*
T0*
_output_shapes
:и
 gradients/Squeeze_1_grad/ReshapeReshapegradients/grad_ys_3:output:0'gradients/Squeeze_1_grad/Shape:output:0*
T0*+
_output_shapes
:         @┼
gradients/AddNAddNgradients/grad_ys_0:output:0,gradients/ExpandDims_2_grad/Reshape:output:0*
N*
T0*&
_class
loc:@gradients/grad_ys_0*'
_output_shapes
:         @}
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:Ж
3gradients/strided_slice_grad/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
         {
1gradients/strided_slice_grad/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 
5gradients/strided_slice_grad/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:в
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0<gradients/strided_slice_grad/StridedSliceGrad/begin:output:0:gradients/strided_slice_grad/StridedSliceGrad/end:output:0>gradients/strided_slice_grad/StridedSliceGrad/strides:output:0gradients/AddN:sum:0*
Index0*
T0*4
_output_shapes"
 :                  @*
shrink_axis_maskc
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
:м
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_11gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnn'gradients_squeeze_1_grad_shape_cudnnrnn6gradients/strided_slice_grad/StridedSliceGrad:output:0'gradients/Squeeze_grad/Reshape:output:0)gradients/Squeeze_1_grad/Reshape:output:0gradients_zeros_like_cudnnrnn*
T0*j
_output_shapesX
V:                  	:         @:         @:АЦЦ
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:┘
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*4
_output_shapes"
 :                  	u
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:┼
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*'
_output_shapes
:         @y
!gradients/ExpandDims_1_grad/ShapeShape(gradients_expanddims_1_grad_shape_init_c*
T0*
_output_shapes
:╔
#gradients/ExpandDims_1_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_c_backprop:0*gradients/ExpandDims_1_grad/Shape:output:0*
T0*'
_output_shapes
:         @^
gradients/concat_1_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :Ъ
gradients/concat_1_grad/modFloorMod)gradients_concat_1_grad_mod_concat_1_axis%gradients/concat_1_grad/Rank:output:0*
T0*
_output_shapes
: h
gradients/concat_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:└j
gradients/concat_1_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:└j
gradients/concat_1_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:└j
gradients/concat_1_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:└j
gradients/concat_1_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:А j
gradients/concat_1_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:А j
gradients/concat_1_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB:А j
gradients/concat_1_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB:А i
gradients/concat_1_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB:@i
gradients/concat_1_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB:@j
 gradients/concat_1_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB:@j
 gradients/concat_1_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB:@j
 gradients/concat_1_grad/Shape_12Const*
_output_shapes
:*
dtype0*
valueB:@j
 gradients/concat_1_grad/Shape_13Const*
_output_shapes
:*
dtype0*
valueB:@j
 gradients/concat_1_grad/Shape_14Const*
_output_shapes
:*
dtype0*
valueB:@j
 gradients/concat_1_grad/Shape_15Const*
_output_shapes
:*
dtype0*
valueB:@°
$gradients/concat_1_grad/ConcatOffsetConcatOffsetgradients/concat_1_grad/mod:z:0&gradients/concat_1_grad/Shape:output:0(gradients/concat_1_grad/Shape_1:output:0(gradients/concat_1_grad/Shape_2:output:0(gradients/concat_1_grad/Shape_3:output:0(gradients/concat_1_grad/Shape_4:output:0(gradients/concat_1_grad/Shape_5:output:0(gradients/concat_1_grad/Shape_6:output:0(gradients/concat_1_grad/Shape_7:output:0(gradients/concat_1_grad/Shape_8:output:0(gradients/concat_1_grad/Shape_9:output:0)gradients/concat_1_grad/Shape_10:output:0)gradients/concat_1_grad/Shape_11:output:0)gradients/concat_1_grad/Shape_12:output:0)gradients/concat_1_grad/Shape_13:output:0)gradients/concat_1_grad/Shape_14:output:0)gradients/concat_1_grad/Shape_15:output:0*
N*t
_output_shapesb
`::::::::::::::::ь
gradients/concat_1_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:0&gradients/concat_1_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:└Ё
gradients/concat_1_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:1(gradients/concat_1_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:└Ё
gradients/concat_1_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:2(gradients/concat_1_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:└Ё
gradients/concat_1_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:3(gradients/concat_1_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:└Ё
gradients/concat_1_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:4(gradients/concat_1_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:А Ё
gradients/concat_1_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:5(gradients/concat_1_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:А Ё
gradients/concat_1_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:6(gradients/concat_1_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes	
:А Ё
gradients/concat_1_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:7(gradients/concat_1_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes	
:А я
gradients/concat_1_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:8(gradients/concat_1_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
:@я
gradients/concat_1_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:9(gradients/concat_1_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
:@Є
 gradients/concat_1_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:10)gradients/concat_1_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
:@Є
 gradients/concat_1_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:11)gradients/concat_1_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
:@Є
 gradients/concat_1_grad/Slice_12Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:12)gradients/concat_1_grad/Shape_12:output:0*
Index0*
T0*
_output_shapes
:@Є
 gradients/concat_1_grad/Slice_13Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:13)gradients/concat_1_grad/Shape_13:output:0*
Index0*
T0*
_output_shapes
:@Є
 gradients/concat_1_grad/Slice_14Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:14)gradients/concat_1_grad/Shape_14:output:0*
Index0*
T0*
_output_shapes
:@Є
 gradients/concat_1_grad/Slice_15Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:15)gradients/concat_1_grad/Shape_15:output:0*
Index0*
T0*
_output_shapes
:@m
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"@   	   б
gradients/Reshape_grad/ReshapeReshape&gradients/concat_1_grad/Slice:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes

:@	o
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"@   	   з
 gradients/Reshape_1_grad/ReshapeReshape(gradients/concat_1_grad/Slice_1:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes

:@	o
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"@   	   з
 gradients/Reshape_2_grad/ReshapeReshape(gradients/concat_1_grad/Slice_2:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes

:@	o
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"@   	   з
 gradients/Reshape_3_grad/ReshapeReshape(gradients/concat_1_grad/Slice_3:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes

:@	o
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"@   @   з
 gradients/Reshape_4_grad/ReshapeReshape(gradients/concat_1_grad/Slice_4:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:@@o
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"@   @   з
 gradients/Reshape_5_grad/ReshapeReshape(gradients/concat_1_grad/Slice_5:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:@@o
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"@   @   з
 gradients/Reshape_6_grad/ReshapeReshape(gradients/concat_1_grad/Slice_6:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:@@o
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"@   @   з
 gradients/Reshape_7_grad/ReshapeReshape(gradients/concat_1_grad/Slice_7:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes

:@@h
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:@г
 gradients/Reshape_8_grad/ReshapeReshape(gradients/concat_1_grad/Slice_8:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
:@h
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:@г
 gradients/Reshape_9_grad/ReshapeReshape(gradients/concat_1_grad/Slice_9:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
:@i
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:@ж
!gradients/Reshape_10_grad/ReshapeReshape)gradients/concat_1_grad/Slice_10:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
:@i
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:@ж
!gradients/Reshape_11_grad/ReshapeReshape)gradients/concat_1_grad/Slice_11:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
:@i
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:@ж
!gradients/Reshape_12_grad/ReshapeReshape)gradients/concat_1_grad/Slice_12:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
:@i
gradients/Reshape_13_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:@ж
!gradients/Reshape_13_grad/ReshapeReshape)gradients/concat_1_grad/Slice_13:output:0(gradients/Reshape_13_grad/Shape:output:0*
T0*
_output_shapes
:@i
gradients/Reshape_14_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:@ж
!gradients/Reshape_14_grad/ReshapeReshape)gradients/concat_1_grad/Slice_14:output:0(gradients/Reshape_14_grad/Shape:output:0*
T0*
_output_shapes
:@i
gradients/Reshape_15_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:@ж
!gradients/Reshape_15_grad/ReshapeReshape)gradients/concat_1_grad/Slice_15:output:0(gradients/Reshape_15_grad/Shape:output:0*
T0*
_output_shapes
:@Ь
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:╡
$gradients/transpose_1_grad/transpose	Transpose'gradients/Reshape_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes

:	@Ь
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:╖
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes

:	@Ь
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:╖
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes

:	@Ь
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:╖
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes

:	@Ь
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:╖
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@@Ь
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:╖
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@@Ь
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:╖
$gradients/transpose_7_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@@Ь
,gradients/transpose_8_grad/InvertPermutationInvertPermutation=gradients_transpose_8_grad_invertpermutation_transpose_8_perm*
_output_shapes
:╖
$gradients/transpose_8_grad/transpose	Transpose)gradients/Reshape_7_grad/Reshape:output:00gradients/transpose_8_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@@ч
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0*gradients/Reshape_13_grad/Reshape:output:0*gradients/Reshape_14_grad/Reshape:output:0*gradients/Reshape_15_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:Ап
gradients/split_grad/concatConcatV2(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0*
_output_shapes
:		А╡
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0(gradients/transpose_7_grad/transpose:y:0(gradients/transpose_8_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0*
_output_shapes
:	@А\
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :Т
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: f
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:Аh
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:А╩
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0*
N* 
_output_shapes
::╥
gradients/concat_grad/SliceSlice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:А╓
gradients/concat_grad/Slice_1Slice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:А{
IdentityIdentity&gradients/transpose_grad/transpose:y:0*
T0*4
_output_shapes"
 :                  	t

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0*
T0*'
_output_shapes
:         @v

Identity_2Identity,gradients/ExpandDims_1_grad/Reshape:output:0*
T0*'
_output_shapes
:         @f

Identity_3Identity$gradients/split_grad/concat:output:0*
T0*
_output_shapes
:		Аh

Identity_4Identity&gradients/split_1_grad/concat:output:0*
T0*
_output_shapes
:	@Аd

Identity_5Identity&gradients/concat_grad/Slice_1:output:0*
T0*
_output_shapes	
:А"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*(
_construction_contextkEagerRuntime*И
_input_shapesЎ
є:         @:         @:         @:         @: :         @:         @:         @:                  @::                  	:         @:         @:АЦ::         @:         @: ::::::::: : : : *=
api_implements+)lstm_e5590ddf-237b-46f1-9660-fe95f9321bd6*
api_preferred_deviceGPU*A
forward_function_name(&__forward_gpu_lstm_with_fallback_24525*
go_backwards( *

time_major( :- )
'
_output_shapes
:         @:1-
+
_output_shapes
:         @:-)
'
_output_shapes
:         @:-)
'
_output_shapes
:         @:

_output_shapes
: :-)
'
_output_shapes
:         @:1-
+
_output_shapes
:         @:1-
+
_output_shapes
:         @::6
4
_output_shapes"
 :                  @:	

_output_shapes
:::
6
4
_output_shapes"
 :                  	:1-
+
_output_shapes
:         @:1-
+
_output_shapes
:         @:"

_output_shapes

:АЦ: 

_output_shapes
::-)
'
_output_shapes
:         @:-)
'
_output_shapes
:         @:

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
в
╣
?__inference_lstm_layer_call_and_return_conditional_losses_20683

inputs/
read_readvariableop_resource:		А1
read_1_readvariableop_resource:	@А-
read_2_readvariableop_resource:	А

identity_3ИвRead/ReadVariableOpвRead_1/ReadVariableOpвRead_2/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:         @R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:         @q
Read/ReadVariableOpReadVariableOpread_readvariableop_resource*
_output_shapes
:		А*
dtype0[
IdentityIdentityRead/ReadVariableOp:value:0*
T0*
_output_shapes
:		Аu
Read_1/ReadVariableOpReadVariableOpread_1_readvariableop_resource*
_output_shapes
:	@А*
dtype0_

Identity_1IdentityRead_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	@Аq
Read_2/ReadVariableOpReadVariableOpread_2_readvariableop_resource*
_output_shapes	
:А*
dtype0[

Identity_2IdentityRead_2/ReadVariableOp:value:0*
T0*
_output_shapes	
:А▒
PartitionedCallPartitionedCallinputszeros:output:0zeros_1:output:0Identity:output:0Identity_1:output:0Identity_2:output:0*
Tin

2*
Tout	
2*
_collective_manager_ids
 *f
_output_shapesT
R:         @:         @:         @:         @: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *(
f#R!
__inference_standard_lstm_20410i

Identity_3IdentityPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         @М
NoOpNoOp^Read/ReadVariableOp^Read_1/ReadVariableOp^Read_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  	: : : 2*
Read/ReadVariableOpRead/ReadVariableOp2.
Read_1/ReadVariableOpRead_1/ReadVariableOp2.
Read_2/ReadVariableOpRead_2/ReadVariableOp:\ X
4
_output_shapes"
 :                  	
 
_user_specified_nameinputs
в
╣
?__inference_lstm_layer_call_and_return_conditional_losses_24528

inputs/
read_readvariableop_resource:		А1
read_1_readvariableop_resource:	@А-
read_2_readvariableop_resource:	А

identity_3ИвRead/ReadVariableOpвRead_1/ReadVariableOpвRead_2/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:         @R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:         @q
Read/ReadVariableOpReadVariableOpread_readvariableop_resource*
_output_shapes
:		А*
dtype0[
IdentityIdentityRead/ReadVariableOp:value:0*
T0*
_output_shapes
:		Аu
Read_1/ReadVariableOpReadVariableOpread_1_readvariableop_resource*
_output_shapes
:	@А*
dtype0_

Identity_1IdentityRead_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	@Аq
Read_2/ReadVariableOpReadVariableOpread_2_readvariableop_resource*
_output_shapes	
:А*
dtype0[

Identity_2IdentityRead_2/ReadVariableOp:value:0*
T0*
_output_shapes	
:А▒
PartitionedCallPartitionedCallinputszeros:output:0zeros_1:output:0Identity:output:0Identity_1:output:0Identity_2:output:0*
Tin

2*
Tout	
2*
_collective_manager_ids
 *f
_output_shapesT
R:         @:         @:         @:         @: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *(
f#R!
__inference_standard_lstm_24255i

Identity_3IdentityPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         @М
NoOpNoOp^Read/ReadVariableOp^Read_1/ReadVariableOp^Read_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  	: : : 2*
Read/ReadVariableOpRead/ReadVariableOp2.
Read_1/ReadVariableOpRead_1/ReadVariableOp2.
Read_2/ReadVariableOpRead_2/ReadVariableOp:\ X
4
_output_shapes"
 :                  	
 
_user_specified_nameinputs
╗;
┐
__inference_standard_lstm_23826

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :                  	B
ShapeShapetranspose:y:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ▓
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    	   р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:щ
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         	*
shrink_axis_maske
MatMulMatMulstrided_slice_1:output:0kernel*
T0*(
_output_shapes
:         А_
MatMul_1MatMulinit_hrecurrent_kernel*
T0*(
_output_shapes
:         Аe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         АT
BiasAddBiasAddadd:z:0bias*
T0*(
_output_shapes
:         АQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :╢
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:         @:         @:         @:         @*
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         @V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         @S
mulMulSigmoid_1:y:0init_c*
T0*'
_output_shapes
:         @N
TanhTanhsplit:output:2*
T0*'
_output_shapes
:         @U
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:         @T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         @V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         @K
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:         @Y
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:         @n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :┼
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ┤
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hinit_cstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelrecurrent_kernelbias*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*c
_output_shapesQ
O: : : : :         @:         @: : :		А:	@А:А* 
_read_only_resource_inputs
 *
_stateful_parallelism( *
bodyR
while_body_23740*
condR
while_cond_23739*b
output_shapesQ
O: : : : :         @:         @: : :		А:	@А:А*
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   ╓
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         @*
element_dtype0*
num_elementsh
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:З
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         @*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ц
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:         @[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  А?`
IdentityIdentitystrided_slice_2:output:0*
T0*'
_output_shapes
:         @]

Identity_1Identitytranspose_1:y:0*
T0*+
_output_shapes
:         @X

Identity_2Identitywhile:output:4*
T0*'
_output_shapes
:         @X

Identity_3Identitywhile:output:5*
T0*'
_output_shapes
:         @I

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:                  	:         @:         @:		А:	@А:А*=
api_implements+)lstm_5a52159d-dcb6-440d-9622-97ba434935b9*
api_preferred_deviceCPU*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :                  	
 
_user_specified_nameinputs:OK
'
_output_shapes
:         @
 
_user_specified_nameinit_h:OK
'
_output_shapes
:         @
 
_user_specified_nameinit_c:GC

_output_shapes
:		А
 
_user_specified_namekernel:QM

_output_shapes
:	@А
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:А

_user_specified_namebias
╗;
┐
__inference_standard_lstm_21795

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :                  	B
ShapeShapetranspose:y:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ▓
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    	   р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:щ
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         	*
shrink_axis_maske
MatMulMatMulstrided_slice_1:output:0kernel*
T0*(
_output_shapes
:         А_
MatMul_1MatMulinit_hrecurrent_kernel*
T0*(
_output_shapes
:         Аe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         АT
BiasAddBiasAddadd:z:0bias*
T0*(
_output_shapes
:         АQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :╢
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:         @:         @:         @:         @*
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         @V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         @S
mulMulSigmoid_1:y:0init_c*
T0*'
_output_shapes
:         @N
TanhTanhsplit:output:2*
T0*'
_output_shapes
:         @U
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:         @T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         @V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         @K
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:         @Y
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:         @n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :┼
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ┤
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hinit_cstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelrecurrent_kernelbias*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*c
_output_shapesQ
O: : : : :         @:         @: : :		А:	@А:А* 
_read_only_resource_inputs
 *
_stateful_parallelism( *
bodyR
while_body_21709*
condR
while_cond_21708*b
output_shapesQ
O: : : : :         @:         @: : :		А:	@А:А*
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   ╓
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         @*
element_dtype0*
num_elementsh
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:З
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         @*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ц
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:         @[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  А?`
IdentityIdentitystrided_slice_2:output:0*
T0*'
_output_shapes
:         @]

Identity_1Identitytranspose_1:y:0*
T0*+
_output_shapes
:         @X

Identity_2Identitywhile:output:4*
T0*'
_output_shapes
:         @X

Identity_3Identitywhile:output:5*
T0*'
_output_shapes
:         @I

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:                  	:         @:         @:		А:	@А:А*=
api_implements+)lstm_d4300e95-cf4b-4bbb-ba38-7f8ecfb78490*
api_preferred_deviceCPU*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :                  	
 
_user_specified_nameinputs:OK
'
_output_shapes
:         @
 
_user_specified_nameinit_h:OK
'
_output_shapes
:         @
 
_user_specified_nameinit_c:GC

_output_shapes
:		А
 
_user_specified_namekernel:QM

_output_shapes
:	@А
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:А

_user_specified_namebias
Ш

У
#__inference_signature_wrapper_22230
normalization_input
unknown
	unknown_0
	unknown_1:		А
	unknown_2:	@А
	unknown_3:	А
	unknown_4:	@э
	unknown_5:	э
identityИвStatefulPartitionedCallВ
StatefulPartitionedCallStatefulPartitionedCallnormalization_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         э*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8В *)
f$R"
 __inference__wrapped_model_20250p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         э`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:'                           :	:	: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:r n
=
_output_shapes+
):'                           
-
_user_specified_namenormalization_input:($
"
_output_shapes
:	:($
"
_output_shapes
:	
╪)
╬
while_body_23311
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_bias_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel#
while_matmul_1_recurrent_kernel
while_biasadd_biasИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    	   ж
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         	*
element_dtype0Т
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*(
_output_shapes
:         АГ
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*(
_output_shapes
:         Аw
	while/addAddV2while/MatMul:product:0while/MatMul_1:product:0*
T0*(
_output_shapes
:         Аp
while/BiasAddBiasAddwhile/add:z:0while_biasadd_bias_0*
T0*(
_output_shapes
:         АW
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :╚
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*`
_output_shapesN
L:         @:         @:         @:         @*
	num_split`
while/SigmoidSigmoidwhile/split:output:0*
T0*'
_output_shapes
:         @b
while/Sigmoid_1Sigmoidwhile/split:output:1*
T0*'
_output_shapes
:         @l
	while/mulMulwhile/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:         @Z

while/TanhTanhwhile/split:output:2*
T0*'
_output_shapes
:         @g
while/mul_1Mulwhile/Sigmoid:y:0while/Tanh:y:0*
T0*'
_output_shapes
:         @f
while/add_1AddV2while/mul:z:0while/mul_1:z:0*
T0*'
_output_shapes
:         @b
while/Sigmoid_2Sigmoidwhile/split:output:3*
T0*'
_output_shapes
:         @W
while/Tanh_1Tanhwhile/add_1:z:0*
T0*'
_output_shapes
:         @k
while/mul_2Mulwhile/Sigmoid_2:y:0while/Tanh_1:y:0*
T0*'
_output_shapes
:         @r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : р
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/mul_2:z:0*
_output_shapes
: *
element_dtype0:щш╥O
while/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :`
while/add_2AddV2while_placeholderwhile/add_2/y:output:0*
T0*
_output_shapes
: O
while/add_3/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_3AddV2while_while_loop_counterwhile/add_3/y:output:0*
T0*
_output_shapes
: L
while/IdentityIdentitywhile/add_3:z:0*
T0*
_output_shapes
: ]
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: N
while/Identity_2Identitywhile/add_2:z:0*
T0*
_output_shapes
: y
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: _
while/Identity_4Identitywhile/mul_2:z:0*
T0*'
_output_shapes
:         @_
while/Identity_5Identitywhile/add_1:z:0*
T0*'
_output_shapes
:         @"*
while_biasadd_biaswhile_biasadd_bias_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O: : : : :         @:         @: : :		А:	@А:А: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         @:-)
'
_output_shapes
:         @:

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:		А:%	!

_output_shapes
:	@А:!


_output_shapes	
:А
╪)
╬
while_body_21215
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_bias_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel#
while_matmul_1_recurrent_kernel
while_biasadd_biasИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    	   ж
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         	*
element_dtype0Т
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*(
_output_shapes
:         АГ
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*(
_output_shapes
:         Аw
	while/addAddV2while/MatMul:product:0while/MatMul_1:product:0*
T0*(
_output_shapes
:         Аp
while/BiasAddBiasAddwhile/add:z:0while_biasadd_bias_0*
T0*(
_output_shapes
:         АW
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :╚
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*`
_output_shapesN
L:         @:         @:         @:         @*
	num_split`
while/SigmoidSigmoidwhile/split:output:0*
T0*'
_output_shapes
:         @b
while/Sigmoid_1Sigmoidwhile/split:output:1*
T0*'
_output_shapes
:         @l
	while/mulMulwhile/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:         @Z

while/TanhTanhwhile/split:output:2*
T0*'
_output_shapes
:         @g
while/mul_1Mulwhile/Sigmoid:y:0while/Tanh:y:0*
T0*'
_output_shapes
:         @f
while/add_1AddV2while/mul:z:0while/mul_1:z:0*
T0*'
_output_shapes
:         @b
while/Sigmoid_2Sigmoidwhile/split:output:3*
T0*'
_output_shapes
:         @W
while/Tanh_1Tanhwhile/add_1:z:0*
T0*'
_output_shapes
:         @k
while/mul_2Mulwhile/Sigmoid_2:y:0while/Tanh_1:y:0*
T0*'
_output_shapes
:         @r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : р
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/mul_2:z:0*
_output_shapes
: *
element_dtype0:щш╥O
while/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :`
while/add_2AddV2while_placeholderwhile/add_2/y:output:0*
T0*
_output_shapes
: O
while/add_3/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_3AddV2while_while_loop_counterwhile/add_3/y:output:0*
T0*
_output_shapes
: L
while/IdentityIdentitywhile/add_3:z:0*
T0*
_output_shapes
: ]
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: N
while/Identity_2Identitywhile/add_2:z:0*
T0*
_output_shapes
: y
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: _
while/Identity_4Identitywhile/mul_2:z:0*
T0*'
_output_shapes
:         @_
while/Identity_5Identitywhile/add_1:z:0*
T0*'
_output_shapes
:         @"*
while_biasadd_biaswhile_biasadd_bias_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O: : : : :         @:         @: : :		А:	@А:А: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         @:-)
'
_output_shapes
:         @:

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:		А:%	!

_output_shapes
:	@А:!


_output_shapes	
:А
к
ы
E__inference_sequential_layer_call_and_return_conditional_losses_21599

inputs
normalization_sub_y
normalization_sqrt_x

lstm_21575:		А

lstm_21577:	@А

lstm_21579:	А
dense_21593:	@э
dense_21595:	э
identityИвdense/StatefulPartitionedCallвlstm/StatefulPartitionedCallt
normalization/subSubinputsnormalization_sub_y*
T0*4
_output_shapes"
 :                  	]
normalization/SqrtSqrtnormalization_sqrt_x*
T0*"
_output_shapes
:	\
normalization/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *Х┐╓3З
normalization/MaximumMaximumnormalization/Sqrt:y:0 normalization/Maximum/y:output:0*
T0*"
_output_shapes
:	С
normalization/truedivRealDivnormalization/sub:z:0normalization/Maximum:z:0*
T0*4
_output_shapes"
 :                  	■
lstm/StatefulPartitionedCallStatefulPartitionedCallnormalization/truediv:z:0
lstm_21575
lstm_21577
lstm_21579*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *H
fCRA
?__inference_lstm_layer_call_and_return_conditional_losses_21574Б
dense/StatefulPartitionedCallStatefulPartitionedCall%lstm/StatefulPartitionedCall:output:0dense_21593dense_21595*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         э*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_21592v
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         эЕ
NoOpNoOp^dense/StatefulPartitionedCall^lstm/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:'                           :	:	: : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2<
lstm/StatefulPartitionedCalllstm/StatefulPartitionedCall:e a
=
_output_shapes+
):'                           
 
_user_specified_nameinputs:($
"
_output_shapes
:	:($
"
_output_shapes
:	
╗;
┐
__inference_standard_lstm_22476

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :                  	B
ShapeShapetranspose:y:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ▓
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    	   р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:щ
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         	*
shrink_axis_maske
MatMulMatMulstrided_slice_1:output:0kernel*
T0*(
_output_shapes
:         А_
MatMul_1MatMulinit_hrecurrent_kernel*
T0*(
_output_shapes
:         Аe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         АT
BiasAddBiasAddadd:z:0bias*
T0*(
_output_shapes
:         АQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :╢
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:         @:         @:         @:         @*
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         @V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         @S
mulMulSigmoid_1:y:0init_c*
T0*'
_output_shapes
:         @N
TanhTanhsplit:output:2*
T0*'
_output_shapes
:         @U
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:         @T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         @V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         @K
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:         @Y
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:         @n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :┼
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ┤
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hinit_cstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelrecurrent_kernelbias*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*c
_output_shapesQ
O: : : : :         @:         @: : :		А:	@А:А* 
_read_only_resource_inputs
 *
_stateful_parallelism( *
bodyR
while_body_22390*
condR
while_cond_22389*b
output_shapesQ
O: : : : :         @:         @: : :		А:	@А:А*
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   ╓
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         @*
element_dtype0*
num_elementsh
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:З
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         @*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ц
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:         @[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  А?`
IdentityIdentitystrided_slice_2:output:0*
T0*'
_output_shapes
:         @]

Identity_1Identitytranspose_1:y:0*
T0*+
_output_shapes
:         @X

Identity_2Identitywhile:output:4*
T0*'
_output_shapes
:         @X

Identity_3Identitywhile:output:5*
T0*'
_output_shapes
:         @I

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:                  	:         @:         @:		А:	@А:А*=
api_implements+)lstm_b4638b8b-6e40-4922-ab41-0ce5301502aa*
api_preferred_deviceCPU*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :                  	
 
_user_specified_nameinputs:OK
'
_output_shapes
:         @
 
_user_specified_nameinit_h:OK
'
_output_shapes
:         @
 
_user_specified_nameinit_c:GC

_output_shapes
:		А
 
_user_specified_namekernel:QM

_output_shapes
:	@А
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:А

_user_specified_namebias
М	
╝
while_cond_20763
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice3
/while_while_cond_20763___redundant_placeholder03
/while_while_cond_20763___redundant_placeholder13
/while_while_cond_20763___redundant_placeholder23
/while_while_cond_20763___redundant_placeholder3
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :         @:         @: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         @:-)
'
_output_shapes
:         @:

_output_shapes
: :

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
М	
╝
while_cond_23739
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice3
/while_while_cond_23739___redundant_placeholder03
/while_while_cond_23739___redundant_placeholder13
/while_while_cond_23739___redundant_placeholder23
/while_while_cond_23739___redundant_placeholder3
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :         @:         @: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         @:-)
'
_output_shapes
:         @:

_output_shapes
: :

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
°)
║
E__inference_sequential_layer_call_and_return_conditional_losses_23197

inputs
normalization_sub_y
normalization_sqrt_x4
!lstm_read_readvariableop_resource:		А6
#lstm_read_1_readvariableop_resource:	@А2
#lstm_read_2_readvariableop_resource:	А7
$dense_matmul_readvariableop_resource:	@э4
%dense_biasadd_readvariableop_resource:	э
identityИвdense/BiasAdd/ReadVariableOpвdense/MatMul/ReadVariableOpвlstm/Read/ReadVariableOpвlstm/Read_1/ReadVariableOpвlstm/Read_2/ReadVariableOpt
normalization/subSubinputsnormalization_sub_y*
T0*4
_output_shapes"
 :                  	]
normalization/SqrtSqrtnormalization_sqrt_x*
T0*"
_output_shapes
:	\
normalization/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *Х┐╓3З
normalization/MaximumMaximumnormalization/Sqrt:y:0 normalization/Maximum/y:output:0*
T0*"
_output_shapes
:	С
normalization/truedivRealDivnormalization/sub:z:0normalization/Maximum:z:0*
T0*4
_output_shapes"
 :                  	S

lstm/ShapeShapenormalization/truediv:z:0*
T0*
_output_shapes
:b
lstm/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: d
lstm/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:d
lstm/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ъ
lstm/strided_sliceStridedSlicelstm/Shape:output:0!lstm/strided_slice/stack:output:0#lstm/strided_slice/stack_1:output:0#lstm/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskU
lstm/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@В
lstm/zeros/packedPacklstm/strided_slice:output:0lstm/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:U
lstm/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    {

lstm/zerosFilllstm/zeros/packed:output:0lstm/zeros/Const:output:0*
T0*'
_output_shapes
:         @W
lstm/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@Ж
lstm/zeros_1/packedPacklstm/strided_slice:output:0lstm/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:W
lstm/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Б
lstm/zeros_1Filllstm/zeros_1/packed:output:0lstm/zeros_1/Const:output:0*
T0*'
_output_shapes
:         @{
lstm/Read/ReadVariableOpReadVariableOp!lstm_read_readvariableop_resource*
_output_shapes
:		А*
dtype0e
lstm/IdentityIdentity lstm/Read/ReadVariableOp:value:0*
T0*
_output_shapes
:		А
lstm/Read_1/ReadVariableOpReadVariableOp#lstm_read_1_readvariableop_resource*
_output_shapes
:	@А*
dtype0i
lstm/Identity_1Identity"lstm/Read_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	@А{
lstm/Read_2/ReadVariableOpReadVariableOp#lstm_read_2_readvariableop_resource*
_output_shapes	
:А*
dtype0e
lstm/Identity_2Identity"lstm/Read_2/ReadVariableOp:value:0*
T0*
_output_shapes	
:Ат
lstm/PartitionedCallPartitionedCallnormalization/truediv:z:0lstm/zeros:output:0lstm/zeros_1:output:0lstm/Identity:output:0lstm/Identity_1:output:0lstm/Identity_2:output:0*
Tin

2*
Tout	
2*
_collective_manager_ids
 *f
_output_shapesT
R:         @:         @:         @:         @: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *(
f#R!
__inference_standard_lstm_22918Б
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	@э*
dtype0Н
dense/MatMulMatMullstm/PartitionedCall:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         э
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:э*
dtype0Й
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         эf
IdentityIdentitydense/BiasAdd:output:0^NoOp*
T0*(
_output_shapes
:         э╪
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^lstm/Read/ReadVariableOp^lstm/Read_1/ReadVariableOp^lstm/Read_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:'                           :	:	: : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp24
lstm/Read/ReadVariableOplstm/Read/ReadVariableOp28
lstm/Read_1/ReadVariableOplstm/Read_1/ReadVariableOp28
lstm/Read_2/ReadVariableOplstm/Read_2/ReadVariableOp:e a
=
_output_shapes+
):'                           
 
_user_specified_nameinputs:($
"
_output_shapes
:	:($
"
_output_shapes
:	
╟@
╦
(__inference_gpu_lstm_with_fallback_23491

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4Иc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :                  	P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : o

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*+
_output_shapes
:         @R
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : s
ExpandDims_1
ExpandDimsinit_cExpandDims_1/dim:output:0*
T0*+
_output_shapes
:         @Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :И
splitSplitsplit/split_dim:output:0kernel*
T0*<
_output_shapes*
(:	@:	@:	@:	@*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ц
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*<
_output_shapes*
(:@@:@@:@@:@@*
	num_splitY

zeros_likeConst*
_output_shapes	
:А*
dtype0*
valueBА*    M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : r
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes	
:АS
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Э
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*D
_output_shapes2
0:@:@:@:@:@:@:@:@*
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
         a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes

:@	Y
ReshapeReshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:└a
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes

:@	[
	Reshape_1Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:└a
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

:@	[
	Reshape_2Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:└a
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes

:@	[
	Reshape_3Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:└a
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:@@[
	Reshape_4Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:А a
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0*
_output_shapes

:@@[
	Reshape_5Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:А a
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0*
_output_shapes

:@@[
	Reshape_6Reshapetranspose_7:y:0Const:output:0*
T0*
_output_shapes	
:А a
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0*
_output_shapes

:@@[
	Reshape_7Reshapetranspose_8:y:0Const:output:0*
T0*
_output_shapes	
:А [
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
:@[
	Reshape_9Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
:@\

Reshape_10Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
:@\

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
:@\

Reshape_12Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
:@\

Reshape_13Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
:@\

Reshape_14Reshapesplit_2:output:6Const:output:0*
T0*
_output_shapes
:@\

Reshape_15Reshapesplit_2:output:7Const:output:0*
T0*
_output_shapes
:@O
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : а
concat_1ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes

:АЦ╙
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1:output:0*
T0*f
_output_shapesT
R:                  @:         @:         @:f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
         _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:х
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         @*
shrink_axis_maskp
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:         @*
squeeze_dims
 r
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*'
_output_shapes
:         @*
squeeze_dims
 R
ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B :Г
ExpandDims_2
ExpandDimsstrided_slice:output:0ExpandDims_2/dim:output:0*
T0*+
_output_shapes
:         @[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @^
IdentityIdentitystrided_slice:output:0*
T0*'
_output_shapes
:         @c

Identity_1IdentityExpandDims_2:output:0*
T0*+
_output_shapes
:         @Z

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:         @\

Identity_3IdentitySqueeze_1:output:0*
T0*'
_output_shapes
:         @I

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:                  	:         @:         @:		А:	@А:А*=
api_implements+)lstm_d60d0f35-f731-4def-8ea6-8cb0eeff7cf2*
api_preferred_deviceGPU*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :                  	
 
_user_specified_nameinputs:OK
'
_output_shapes
:         @
 
_user_specified_nameinit_h:OK
'
_output_shapes
:         @
 
_user_specified_nameinit_c:GC

_output_shapes
:		А
 
_user_specified_namekernel:QM

_output_shapes
:	@А
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:А

_user_specified_namebias
ёK
а
&__forward_gpu_lstm_with_fallback_21120

inputs
init_h_0
init_c_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4
strided_slice
cudnnrnn

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
expanddims_1
concat_1
transpose_perm

init_h

init_c
concat_1_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
transpose_7_perm
transpose_8_perm
split_2_split_dim
split_split_dim
split_1_split_dim
concat_axisИc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          B
transpose_0	Transposeinputstranspose/perm:output:0*
T0P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : q

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0*+
_output_shapes
:         @R
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : u
ExpandDims_1
ExpandDimsinit_c_0ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:         @Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :И
splitSplitsplit/split_dim:output:0kernel*
T0*<
_output_shapes*
(:	@:	@:	@:	@*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ц
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*<
_output_shapes*
(:@@:@@:@@:@@*
	num_splitY

zeros_likeConst*
_output_shapes	
:А*
dtype0*
valueBА*    M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : r
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes	
:АS
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Э
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*D
_output_shapes2
0:@:@:@:@:@:@:@:@*
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
         a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes

:@	Y
ReshapeReshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:└a
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes

:@	[
	Reshape_1Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:└a
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

:@	[
	Reshape_2Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:└a
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes

:@	[
	Reshape_3Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:└a
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:@@[
	Reshape_4Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:А a
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0*
_output_shapes

:@@[
	Reshape_5Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:А a
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0*
_output_shapes

:@@[
	Reshape_6Reshapetranspose_7:y:0Const:output:0*
T0*
_output_shapes	
:А a
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0*
_output_shapes

:@@[
	Reshape_7Reshapetranspose_8:y:0Const:output:0*
T0*
_output_shapes	
:А [
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
:@[
	Reshape_9Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
:@\

Reshape_10Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
:@\

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
:@\

Reshape_12Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
:@\

Reshape_13Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
:@\

Reshape_14Reshapesplit_2:output:6Const:output:0*
T0*
_output_shapes
:@\

Reshape_15Reshapesplit_2:output:7Const:output:0*
T0*
_output_shapes
:@O
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Д

concat_1_0ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0╫
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1_0:output:0*
T0*f
_output_shapesT
R:                  @:         @:         @:f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
         _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╛
strided_slice_0StridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_maskp
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:         @*
squeeze_dims
 r
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*'
_output_shapes
:         @*
squeeze_dims
 R
ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B :Е
ExpandDims_2
ExpandDimsstrided_slice_0:output:0ExpandDims_2/dim:output:0*
T0*+
_output_shapes
:         @[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @`
IdentityIdentitystrided_slice_0:output:0*
T0*'
_output_shapes
:         @c

Identity_1IdentityExpandDims_2:output:0*
T0*+
_output_shapes
:         @Z

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:         @\

Identity_3IdentitySqueeze_1:output:0*
T0*'
_output_shapes
:         @I

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
concat_1concat_1_0:output:0"'
concat_1_axisconcat_1/axis:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output_h:0"!

cudnnrnn_0CudnnRNN:output_c:0"

cudnnrnn_1CudnnRNN:output:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"!

expanddimsExpandDims:output:0"%
expanddims_1ExpandDims_1:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
init_cinit_c_0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0")
strided_slicestrided_slice_0:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0"-
transpose_7_permtranspose_7/perm:output:0"-
transpose_8_permtranspose_8/perm:output:0")
transpose_permtranspose/perm:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:                  	:         @:         @:		А:	@А:А*=
api_implements+)lstm_733a4ab0-824c-43e0-83a3-926a8e9369c1*
api_preferred_deviceGPU*U
backward_function_name;9__inference___backward_gpu_lstm_with_fallback_20945_21121*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :                  	
 
_user_specified_nameinputs:OK
'
_output_shapes
:         @
 
_user_specified_nameinit_h:OK
'
_output_shapes
:         @
 
_user_specified_nameinit_c:GC

_output_shapes
:		А
 
_user_specified_namekernel:QM

_output_shapes
:	@А
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:А

_user_specified_namebias
Г┬
у
9__inference___backward_gpu_lstm_with_fallback_21396_21572
placeholder
placeholder_1
placeholder_2
placeholder_3
placeholder_43
/gradients_expanddims_2_grad_shape_strided_slice)
%gradients_squeeze_grad_shape_cudnnrnn+
'gradients_squeeze_1_grad_shape_cudnnrnn/
+gradients_strided_slice_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9
5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_15
1gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h,
(gradients_expanddims_1_grad_shape_init_c-
)gradients_concat_1_grad_mod_concat_1_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_permA
=gradients_transpose_7_grad_invertpermutation_transpose_7_permA
=gradients_transpose_8_grad_invertpermutation_transpose_8_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim)
%gradients_concat_grad_mod_concat_axis
identity

identity_1

identity_2

identity_3

identity_4

identity_5И^
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:         @d
gradients/grad_ys_1Identityplaceholder_1*
T0*+
_output_shapes
:         @`
gradients/grad_ys_2Identityplaceholder_2*
T0*'
_output_shapes
:         @`
gradients/grad_ys_3Identityplaceholder_3*
T0*'
_output_shapes
:         @O
gradients/grad_ys_4Identityplaceholder_4*
T0*
_output_shapes
: А
!gradients/ExpandDims_2_grad/ShapeShape/gradients_expanddims_2_grad_shape_strided_slice*
T0*
_output_shapes
:к
#gradients/ExpandDims_2_grad/ReshapeReshapegradients/grad_ys_1:output:0*gradients/ExpandDims_2_grad/Shape:output:0*
T0*'
_output_shapes
:         @q
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
:д
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*+
_output_shapes
:         @u
gradients/Squeeze_1_grad/ShapeShape'gradients_squeeze_1_grad_shape_cudnnrnn*
T0*
_output_shapes
:и
 gradients/Squeeze_1_grad/ReshapeReshapegradients/grad_ys_3:output:0'gradients/Squeeze_1_grad/Shape:output:0*
T0*+
_output_shapes
:         @┼
gradients/AddNAddNgradients/grad_ys_0:output:0,gradients/ExpandDims_2_grad/Reshape:output:0*
N*
T0*&
_class
loc:@gradients/grad_ys_0*'
_output_shapes
:         @}
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:Ж
3gradients/strided_slice_grad/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
         {
1gradients/strided_slice_grad/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 
5gradients/strided_slice_grad/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:в
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0<gradients/strided_slice_grad/StridedSliceGrad/begin:output:0:gradients/strided_slice_grad/StridedSliceGrad/end:output:0>gradients/strided_slice_grad/StridedSliceGrad/strides:output:0gradients/AddN:sum:0*
Index0*
T0*4
_output_shapes"
 :                  @*
shrink_axis_maskc
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
:м
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_11gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnn'gradients_squeeze_1_grad_shape_cudnnrnn6gradients/strided_slice_grad/StridedSliceGrad:output:0'gradients/Squeeze_grad/Reshape:output:0)gradients/Squeeze_1_grad/Reshape:output:0gradients_zeros_like_cudnnrnn*
T0*j
_output_shapesX
V:                  	:         @:         @:АЦЦ
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:┘
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*4
_output_shapes"
 :                  	u
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:┼
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*'
_output_shapes
:         @y
!gradients/ExpandDims_1_grad/ShapeShape(gradients_expanddims_1_grad_shape_init_c*
T0*
_output_shapes
:╔
#gradients/ExpandDims_1_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_c_backprop:0*gradients/ExpandDims_1_grad/Shape:output:0*
T0*'
_output_shapes
:         @^
gradients/concat_1_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :Ъ
gradients/concat_1_grad/modFloorMod)gradients_concat_1_grad_mod_concat_1_axis%gradients/concat_1_grad/Rank:output:0*
T0*
_output_shapes
: h
gradients/concat_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:└j
gradients/concat_1_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:└j
gradients/concat_1_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:└j
gradients/concat_1_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:└j
gradients/concat_1_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:А j
gradients/concat_1_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:А j
gradients/concat_1_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB:А j
gradients/concat_1_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB:А i
gradients/concat_1_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB:@i
gradients/concat_1_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB:@j
 gradients/concat_1_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB:@j
 gradients/concat_1_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB:@j
 gradients/concat_1_grad/Shape_12Const*
_output_shapes
:*
dtype0*
valueB:@j
 gradients/concat_1_grad/Shape_13Const*
_output_shapes
:*
dtype0*
valueB:@j
 gradients/concat_1_grad/Shape_14Const*
_output_shapes
:*
dtype0*
valueB:@j
 gradients/concat_1_grad/Shape_15Const*
_output_shapes
:*
dtype0*
valueB:@°
$gradients/concat_1_grad/ConcatOffsetConcatOffsetgradients/concat_1_grad/mod:z:0&gradients/concat_1_grad/Shape:output:0(gradients/concat_1_grad/Shape_1:output:0(gradients/concat_1_grad/Shape_2:output:0(gradients/concat_1_grad/Shape_3:output:0(gradients/concat_1_grad/Shape_4:output:0(gradients/concat_1_grad/Shape_5:output:0(gradients/concat_1_grad/Shape_6:output:0(gradients/concat_1_grad/Shape_7:output:0(gradients/concat_1_grad/Shape_8:output:0(gradients/concat_1_grad/Shape_9:output:0)gradients/concat_1_grad/Shape_10:output:0)gradients/concat_1_grad/Shape_11:output:0)gradients/concat_1_grad/Shape_12:output:0)gradients/concat_1_grad/Shape_13:output:0)gradients/concat_1_grad/Shape_14:output:0)gradients/concat_1_grad/Shape_15:output:0*
N*t
_output_shapesb
`::::::::::::::::ь
gradients/concat_1_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:0&gradients/concat_1_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:└Ё
gradients/concat_1_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:1(gradients/concat_1_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:└Ё
gradients/concat_1_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:2(gradients/concat_1_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:└Ё
gradients/concat_1_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:3(gradients/concat_1_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:└Ё
gradients/concat_1_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:4(gradients/concat_1_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:А Ё
gradients/concat_1_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:5(gradients/concat_1_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:А Ё
gradients/concat_1_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:6(gradients/concat_1_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes	
:А Ё
gradients/concat_1_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:7(gradients/concat_1_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes	
:А я
gradients/concat_1_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:8(gradients/concat_1_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
:@я
gradients/concat_1_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:9(gradients/concat_1_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
:@Є
 gradients/concat_1_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:10)gradients/concat_1_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
:@Є
 gradients/concat_1_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:11)gradients/concat_1_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
:@Є
 gradients/concat_1_grad/Slice_12Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:12)gradients/concat_1_grad/Shape_12:output:0*
Index0*
T0*
_output_shapes
:@Є
 gradients/concat_1_grad/Slice_13Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:13)gradients/concat_1_grad/Shape_13:output:0*
Index0*
T0*
_output_shapes
:@Є
 gradients/concat_1_grad/Slice_14Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:14)gradients/concat_1_grad/Shape_14:output:0*
Index0*
T0*
_output_shapes
:@Є
 gradients/concat_1_grad/Slice_15Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:15)gradients/concat_1_grad/Shape_15:output:0*
Index0*
T0*
_output_shapes
:@m
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"@   	   б
gradients/Reshape_grad/ReshapeReshape&gradients/concat_1_grad/Slice:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes

:@	o
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"@   	   з
 gradients/Reshape_1_grad/ReshapeReshape(gradients/concat_1_grad/Slice_1:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes

:@	o
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"@   	   з
 gradients/Reshape_2_grad/ReshapeReshape(gradients/concat_1_grad/Slice_2:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes

:@	o
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"@   	   з
 gradients/Reshape_3_grad/ReshapeReshape(gradients/concat_1_grad/Slice_3:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes

:@	o
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"@   @   з
 gradients/Reshape_4_grad/ReshapeReshape(gradients/concat_1_grad/Slice_4:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:@@o
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"@   @   з
 gradients/Reshape_5_grad/ReshapeReshape(gradients/concat_1_grad/Slice_5:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:@@o
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"@   @   з
 gradients/Reshape_6_grad/ReshapeReshape(gradients/concat_1_grad/Slice_6:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:@@o
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"@   @   з
 gradients/Reshape_7_grad/ReshapeReshape(gradients/concat_1_grad/Slice_7:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes

:@@h
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:@г
 gradients/Reshape_8_grad/ReshapeReshape(gradients/concat_1_grad/Slice_8:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
:@h
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:@г
 gradients/Reshape_9_grad/ReshapeReshape(gradients/concat_1_grad/Slice_9:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
:@i
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:@ж
!gradients/Reshape_10_grad/ReshapeReshape)gradients/concat_1_grad/Slice_10:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
:@i
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:@ж
!gradients/Reshape_11_grad/ReshapeReshape)gradients/concat_1_grad/Slice_11:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
:@i
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:@ж
!gradients/Reshape_12_grad/ReshapeReshape)gradients/concat_1_grad/Slice_12:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
:@i
gradients/Reshape_13_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:@ж
!gradients/Reshape_13_grad/ReshapeReshape)gradients/concat_1_grad/Slice_13:output:0(gradients/Reshape_13_grad/Shape:output:0*
T0*
_output_shapes
:@i
gradients/Reshape_14_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:@ж
!gradients/Reshape_14_grad/ReshapeReshape)gradients/concat_1_grad/Slice_14:output:0(gradients/Reshape_14_grad/Shape:output:0*
T0*
_output_shapes
:@i
gradients/Reshape_15_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:@ж
!gradients/Reshape_15_grad/ReshapeReshape)gradients/concat_1_grad/Slice_15:output:0(gradients/Reshape_15_grad/Shape:output:0*
T0*
_output_shapes
:@Ь
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:╡
$gradients/transpose_1_grad/transpose	Transpose'gradients/Reshape_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes

:	@Ь
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:╖
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes

:	@Ь
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:╖
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes

:	@Ь
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:╖
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes

:	@Ь
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:╖
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@@Ь
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:╖
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@@Ь
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:╖
$gradients/transpose_7_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@@Ь
,gradients/transpose_8_grad/InvertPermutationInvertPermutation=gradients_transpose_8_grad_invertpermutation_transpose_8_perm*
_output_shapes
:╖
$gradients/transpose_8_grad/transpose	Transpose)gradients/Reshape_7_grad/Reshape:output:00gradients/transpose_8_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@@ч
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0*gradients/Reshape_13_grad/Reshape:output:0*gradients/Reshape_14_grad/Reshape:output:0*gradients/Reshape_15_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:Ап
gradients/split_grad/concatConcatV2(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0*
_output_shapes
:		А╡
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0(gradients/transpose_7_grad/transpose:y:0(gradients/transpose_8_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0*
_output_shapes
:	@А\
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :Т
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: f
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:Аh
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:А╩
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0*
N* 
_output_shapes
::╥
gradients/concat_grad/SliceSlice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:А╓
gradients/concat_grad/Slice_1Slice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:А{
IdentityIdentity&gradients/transpose_grad/transpose:y:0*
T0*4
_output_shapes"
 :                  	t

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0*
T0*'
_output_shapes
:         @v

Identity_2Identity,gradients/ExpandDims_1_grad/Reshape:output:0*
T0*'
_output_shapes
:         @f

Identity_3Identity$gradients/split_grad/concat:output:0*
T0*
_output_shapes
:		Аh

Identity_4Identity&gradients/split_1_grad/concat:output:0*
T0*
_output_shapes
:	@Аd

Identity_5Identity&gradients/concat_grad/Slice_1:output:0*
T0*
_output_shapes	
:А"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*(
_construction_contextkEagerRuntime*И
_input_shapesЎ
є:         @:         @:         @:         @: :         @:         @:         @:                  @::                  	:         @:         @:АЦ::         @:         @: ::::::::: : : : *=
api_implements+)lstm_e8470502-1520-4f1d-96c5-a71b9a8e2c85*
api_preferred_deviceGPU*A
forward_function_name(&__forward_gpu_lstm_with_fallback_21571*
go_backwards( *

time_major( :- )
'
_output_shapes
:         @:1-
+
_output_shapes
:         @:-)
'
_output_shapes
:         @:-)
'
_output_shapes
:         @:

_output_shapes
: :-)
'
_output_shapes
:         @:1-
+
_output_shapes
:         @:1-
+
_output_shapes
:         @::6
4
_output_shapes"
 :                  @:	

_output_shapes
:::
6
4
_output_shapes"
 :                  	:1-
+
_output_shapes
:         @:1-
+
_output_shapes
:         @:"

_output_shapes

:АЦ: 

_output_shapes
::-)
'
_output_shapes
:         @:-)
'
_output_shapes
:         @:

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
╥
°
E__inference_sequential_layer_call_and_return_conditional_losses_22180
normalization_input
normalization_sub_y
normalization_sqrt_x

lstm_22167:		А

lstm_22169:	@А

lstm_22171:	А
dense_22174:	@э
dense_22176:	э
identityИвdense/StatefulPartitionedCallвlstm/StatefulPartitionedCallБ
normalization/subSubnormalization_inputnormalization_sub_y*
T0*4
_output_shapes"
 :                  	]
normalization/SqrtSqrtnormalization_sqrt_x*
T0*"
_output_shapes
:	\
normalization/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *Х┐╓3З
normalization/MaximumMaximumnormalization/Sqrt:y:0 normalization/Maximum/y:output:0*
T0*"
_output_shapes
:	С
normalization/truedivRealDivnormalization/sub:z:0normalization/Maximum:z:0*
T0*4
_output_shapes"
 :                  	■
lstm/StatefulPartitionedCallStatefulPartitionedCallnormalization/truediv:z:0
lstm_22167
lstm_22169
lstm_22171*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *H
fCRA
?__inference_lstm_layer_call_and_return_conditional_losses_21574Б
dense/StatefulPartitionedCallStatefulPartitionedCall%lstm/StatefulPartitionedCall:output:0dense_22174dense_22176*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         э*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_21592v
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         эЕ
NoOpNoOp^dense/StatefulPartitionedCall^lstm/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:'                           :	:	: : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2<
lstm/StatefulPartitionedCalllstm/StatefulPartitionedCall:r n
=
_output_shapes+
):'                           
-
_user_specified_namenormalization_input:($
"
_output_shapes
:	:($
"
_output_shapes
:	
╗;
┐
__inference_standard_lstm_19971

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :                  	B
ShapeShapetranspose:y:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ▓
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    	   р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:щ
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         	*
shrink_axis_maske
MatMulMatMulstrided_slice_1:output:0kernel*
T0*(
_output_shapes
:         А_
MatMul_1MatMulinit_hrecurrent_kernel*
T0*(
_output_shapes
:         Аe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         АT
BiasAddBiasAddadd:z:0bias*
T0*(
_output_shapes
:         АQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :╢
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:         @:         @:         @:         @*
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         @V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         @S
mulMulSigmoid_1:y:0init_c*
T0*'
_output_shapes
:         @N
TanhTanhsplit:output:2*
T0*'
_output_shapes
:         @U
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:         @T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         @V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         @K
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:         @Y
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:         @n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :┼
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ┤
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hinit_cstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelrecurrent_kernelbias*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*c
_output_shapesQ
O: : : : :         @:         @: : :		А:	@А:А* 
_read_only_resource_inputs
 *
_stateful_parallelism( *
bodyR
while_body_19885*
condR
while_cond_19884*b
output_shapesQ
O: : : : :         @:         @: : :		А:	@А:А*
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   ╓
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         @*
element_dtype0*
num_elementsh
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:З
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         @*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ц
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:         @[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  А?`
IdentityIdentitystrided_slice_2:output:0*
T0*'
_output_shapes
:         @]

Identity_1Identitytranspose_1:y:0*
T0*+
_output_shapes
:         @X

Identity_2Identitywhile:output:4*
T0*'
_output_shapes
:         @X

Identity_3Identitywhile:output:5*
T0*'
_output_shapes
:         @I

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:                  	:         @:         @:		А:	@А:А*=
api_implements+)lstm_59b087b6-9d64-470e-a94e-2d14718ac25a*
api_preferred_deviceCPU*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :                  	
 
_user_specified_nameinputs:OK
'
_output_shapes
:         @
 
_user_specified_nameinit_h:OK
'
_output_shapes
:         @
 
_user_specified_nameinit_c:GC

_output_shapes
:		А
 
_user_specified_namekernel:QM

_output_shapes
:	@А
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:А

_user_specified_namebias
╗;
┐
__inference_standard_lstm_20850

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :                  	B
ShapeShapetranspose:y:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ▓
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    	   р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:щ
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         	*
shrink_axis_maske
MatMulMatMulstrided_slice_1:output:0kernel*
T0*(
_output_shapes
:         А_
MatMul_1MatMulinit_hrecurrent_kernel*
T0*(
_output_shapes
:         Аe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         АT
BiasAddBiasAddadd:z:0bias*
T0*(
_output_shapes
:         АQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :╢
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:         @:         @:         @:         @*
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         @V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         @S
mulMulSigmoid_1:y:0init_c*
T0*'
_output_shapes
:         @N
TanhTanhsplit:output:2*
T0*'
_output_shapes
:         @U
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:         @T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         @V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         @K
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:         @Y
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:         @n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :┼
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ┤
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hinit_cstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelrecurrent_kernelbias*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*c
_output_shapesQ
O: : : : :         @:         @: : :		А:	@А:А* 
_read_only_resource_inputs
 *
_stateful_parallelism( *
bodyR
while_body_20764*
condR
while_cond_20763*b
output_shapesQ
O: : : : :         @:         @: : :		А:	@А:А*
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   ╓
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         @*
element_dtype0*
num_elementsh
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:З
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         @*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ц
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:         @[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  А?`
IdentityIdentitystrided_slice_2:output:0*
T0*'
_output_shapes
:         @]

Identity_1Identitytranspose_1:y:0*
T0*+
_output_shapes
:         @X

Identity_2Identitywhile:output:4*
T0*'
_output_shapes
:         @X

Identity_3Identitywhile:output:5*
T0*'
_output_shapes
:         @I

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:                  	:         @:         @:		А:	@А:А*=
api_implements+)lstm_733a4ab0-824c-43e0-83a3-926a8e9369c1*
api_preferred_deviceCPU*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :                  	
 
_user_specified_nameinputs:OK
'
_output_shapes
:         @
 
_user_specified_nameinit_h:OK
'
_output_shapes
:         @
 
_user_specified_nameinit_c:GC

_output_shapes
:		А
 
_user_specified_namekernel:QM

_output_shapes
:	@А
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:А

_user_specified_namebias
в
╣
?__inference_lstm_layer_call_and_return_conditional_losses_22068

inputs/
read_readvariableop_resource:		А1
read_1_readvariableop_resource:	@А-
read_2_readvariableop_resource:	А

identity_3ИвRead/ReadVariableOpвRead_1/ReadVariableOpвRead_2/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:         @R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:         @q
Read/ReadVariableOpReadVariableOpread_readvariableop_resource*
_output_shapes
:		А*
dtype0[
IdentityIdentityRead/ReadVariableOp:value:0*
T0*
_output_shapes
:		Аu
Read_1/ReadVariableOpReadVariableOpread_1_readvariableop_resource*
_output_shapes
:	@А*
dtype0_

Identity_1IdentityRead_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	@Аq
Read_2/ReadVariableOpReadVariableOpread_2_readvariableop_resource*
_output_shapes	
:А*
dtype0[

Identity_2IdentityRead_2/ReadVariableOp:value:0*
T0*
_output_shapes	
:А▒
PartitionedCallPartitionedCallinputszeros:output:0zeros_1:output:0Identity:output:0Identity_1:output:0Identity_2:output:0*
Tin

2*
Tout	
2*
_collective_manager_ids
 *f
_output_shapesT
R:         @:         @:         @:         @: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *(
f#R!
__inference_standard_lstm_21795i

Identity_3IdentityPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         @М
NoOpNoOp^Read/ReadVariableOp^Read_1/ReadVariableOp^Read_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  	: : : 2*
Read/ReadVariableOpRead/ReadVariableOp2.
Read_1/ReadVariableOpRead_1/ReadVariableOp2.
Read_2/ReadVariableOpRead_2/ReadVariableOp:\ X
4
_output_shapes"
 :                  	
 
_user_specified_nameinputs
╟@
╦
(__inference_gpu_lstm_with_fallback_22570

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4Иc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :                  	P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : o

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*+
_output_shapes
:         @R
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : s
ExpandDims_1
ExpandDimsinit_cExpandDims_1/dim:output:0*
T0*+
_output_shapes
:         @Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :И
splitSplitsplit/split_dim:output:0kernel*
T0*<
_output_shapes*
(:	@:	@:	@:	@*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ц
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*<
_output_shapes*
(:@@:@@:@@:@@*
	num_splitY

zeros_likeConst*
_output_shapes	
:А*
dtype0*
valueBА*    M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : r
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes	
:АS
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Э
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*D
_output_shapes2
0:@:@:@:@:@:@:@:@*
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
         a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes

:@	Y
ReshapeReshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:└a
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes

:@	[
	Reshape_1Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:└a
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

:@	[
	Reshape_2Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:└a
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes

:@	[
	Reshape_3Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:└a
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:@@[
	Reshape_4Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:А a
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0*
_output_shapes

:@@[
	Reshape_5Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:А a
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0*
_output_shapes

:@@[
	Reshape_6Reshapetranspose_7:y:0Const:output:0*
T0*
_output_shapes	
:А a
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0*
_output_shapes

:@@[
	Reshape_7Reshapetranspose_8:y:0Const:output:0*
T0*
_output_shapes	
:А [
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
:@[
	Reshape_9Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
:@\

Reshape_10Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
:@\

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
:@\

Reshape_12Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
:@\

Reshape_13Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
:@\

Reshape_14Reshapesplit_2:output:6Const:output:0*
T0*
_output_shapes
:@\

Reshape_15Reshapesplit_2:output:7Const:output:0*
T0*
_output_shapes
:@O
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : а
concat_1ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes

:АЦ╙
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1:output:0*
T0*f
_output_shapesT
R:                  @:         @:         @:f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
         _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:х
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         @*
shrink_axis_maskp
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:         @*
squeeze_dims
 r
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*'
_output_shapes
:         @*
squeeze_dims
 R
ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B :Г
ExpandDims_2
ExpandDimsstrided_slice:output:0ExpandDims_2/dim:output:0*
T0*+
_output_shapes
:         @[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @^
IdentityIdentitystrided_slice:output:0*
T0*'
_output_shapes
:         @c

Identity_1IdentityExpandDims_2:output:0*
T0*+
_output_shapes
:         @Z

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:         @\

Identity_3IdentitySqueeze_1:output:0*
T0*'
_output_shapes
:         @I

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:                  	:         @:         @:		А:	@А:А*=
api_implements+)lstm_b4638b8b-6e40-4922-ab41-0ce5301502aa*
api_preferred_deviceGPU*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :                  	
 
_user_specified_nameinputs:OK
'
_output_shapes
:         @
 
_user_specified_nameinit_h:OK
'
_output_shapes
:         @
 
_user_specified_nameinit_c:GC

_output_shapes
:		А
 
_user_specified_namekernel:QM

_output_shapes
:	@А
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:А

_user_specified_namebias
╪)
╬
while_body_24598
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_bias_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel#
while_matmul_1_recurrent_kernel
while_biasadd_biasИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    	   ж
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         	*
element_dtype0Т
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*(
_output_shapes
:         АГ
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*(
_output_shapes
:         Аw
	while/addAddV2while/MatMul:product:0while/MatMul_1:product:0*
T0*(
_output_shapes
:         Аp
while/BiasAddBiasAddwhile/add:z:0while_biasadd_bias_0*
T0*(
_output_shapes
:         АW
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :╚
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*`
_output_shapesN
L:         @:         @:         @:         @*
	num_split`
while/SigmoidSigmoidwhile/split:output:0*
T0*'
_output_shapes
:         @b
while/Sigmoid_1Sigmoidwhile/split:output:1*
T0*'
_output_shapes
:         @l
	while/mulMulwhile/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:         @Z

while/TanhTanhwhile/split:output:2*
T0*'
_output_shapes
:         @g
while/mul_1Mulwhile/Sigmoid:y:0while/Tanh:y:0*
T0*'
_output_shapes
:         @f
while/add_1AddV2while/mul:z:0while/mul_1:z:0*
T0*'
_output_shapes
:         @b
while/Sigmoid_2Sigmoidwhile/split:output:3*
T0*'
_output_shapes
:         @W
while/Tanh_1Tanhwhile/add_1:z:0*
T0*'
_output_shapes
:         @k
while/mul_2Mulwhile/Sigmoid_2:y:0while/Tanh_1:y:0*
T0*'
_output_shapes
:         @r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : р
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/mul_2:z:0*
_output_shapes
: *
element_dtype0:щш╥O
while/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :`
while/add_2AddV2while_placeholderwhile/add_2/y:output:0*
T0*
_output_shapes
: O
while/add_3/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_3AddV2while_while_loop_counterwhile/add_3/y:output:0*
T0*
_output_shapes
: L
while/IdentityIdentitywhile/add_3:z:0*
T0*
_output_shapes
: ]
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: N
while/Identity_2Identitywhile/add_2:z:0*
T0*
_output_shapes
: y
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: _
while/Identity_4Identitywhile/mul_2:z:0*
T0*'
_output_shapes
:         @_
while/Identity_5Identitywhile/add_1:z:0*
T0*'
_output_shapes
:         @"*
while_biasadd_biaswhile_biasadd_bias_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O: : : : :         @:         @: : :		А:	@А:А: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         @:-)
'
_output_shapes
:         @:

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:		А:%	!

_output_shapes
:	@А:!


_output_shapes	
:А
╪)
╬
while_body_22832
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_bias_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel#
while_matmul_1_recurrent_kernel
while_biasadd_biasИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    	   ж
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         	*
element_dtype0Т
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*(
_output_shapes
:         АГ
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*(
_output_shapes
:         Аw
	while/addAddV2while/MatMul:product:0while/MatMul_1:product:0*
T0*(
_output_shapes
:         Аp
while/BiasAddBiasAddwhile/add:z:0while_biasadd_bias_0*
T0*(
_output_shapes
:         АW
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :╚
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*`
_output_shapesN
L:         @:         @:         @:         @*
	num_split`
while/SigmoidSigmoidwhile/split:output:0*
T0*'
_output_shapes
:         @b
while/Sigmoid_1Sigmoidwhile/split:output:1*
T0*'
_output_shapes
:         @l
	while/mulMulwhile/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:         @Z

while/TanhTanhwhile/split:output:2*
T0*'
_output_shapes
:         @g
while/mul_1Mulwhile/Sigmoid:y:0while/Tanh:y:0*
T0*'
_output_shapes
:         @f
while/add_1AddV2while/mul:z:0while/mul_1:z:0*
T0*'
_output_shapes
:         @b
while/Sigmoid_2Sigmoidwhile/split:output:3*
T0*'
_output_shapes
:         @W
while/Tanh_1Tanhwhile/add_1:z:0*
T0*'
_output_shapes
:         @k
while/mul_2Mulwhile/Sigmoid_2:y:0while/Tanh_1:y:0*
T0*'
_output_shapes
:         @r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : р
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/mul_2:z:0*
_output_shapes
: *
element_dtype0:щш╥O
while/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :`
while/add_2AddV2while_placeholderwhile/add_2/y:output:0*
T0*
_output_shapes
: O
while/add_3/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_3AddV2while_while_loop_counterwhile/add_3/y:output:0*
T0*
_output_shapes
: L
while/IdentityIdentitywhile/add_3:z:0*
T0*
_output_shapes
: ]
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: N
while/Identity_2Identitywhile/add_2:z:0*
T0*
_output_shapes
: y
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: _
while/Identity_4Identitywhile/mul_2:z:0*
T0*'
_output_shapes
:         @_
while/Identity_5Identitywhile/add_1:z:0*
T0*'
_output_shapes
:         @"*
while_biasadd_biaswhile_biasadd_bias_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O: : : : :         @:         @: : :		А:	@А:А: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         @:-)
'
_output_shapes
:         @:

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:		А:%	!

_output_shapes
:	@А:!


_output_shapes	
:А
°)
║
E__inference_sequential_layer_call_and_return_conditional_losses_22755

inputs
normalization_sub_y
normalization_sqrt_x4
!lstm_read_readvariableop_resource:		А6
#lstm_read_1_readvariableop_resource:	@А2
#lstm_read_2_readvariableop_resource:	А7
$dense_matmul_readvariableop_resource:	@э4
%dense_biasadd_readvariableop_resource:	э
identityИвdense/BiasAdd/ReadVariableOpвdense/MatMul/ReadVariableOpвlstm/Read/ReadVariableOpвlstm/Read_1/ReadVariableOpвlstm/Read_2/ReadVariableOpt
normalization/subSubinputsnormalization_sub_y*
T0*4
_output_shapes"
 :                  	]
normalization/SqrtSqrtnormalization_sqrt_x*
T0*"
_output_shapes
:	\
normalization/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *Х┐╓3З
normalization/MaximumMaximumnormalization/Sqrt:y:0 normalization/Maximum/y:output:0*
T0*"
_output_shapes
:	С
normalization/truedivRealDivnormalization/sub:z:0normalization/Maximum:z:0*
T0*4
_output_shapes"
 :                  	S

lstm/ShapeShapenormalization/truediv:z:0*
T0*
_output_shapes
:b
lstm/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: d
lstm/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:d
lstm/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ъ
lstm/strided_sliceStridedSlicelstm/Shape:output:0!lstm/strided_slice/stack:output:0#lstm/strided_slice/stack_1:output:0#lstm/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskU
lstm/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@В
lstm/zeros/packedPacklstm/strided_slice:output:0lstm/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:U
lstm/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    {

lstm/zerosFilllstm/zeros/packed:output:0lstm/zeros/Const:output:0*
T0*'
_output_shapes
:         @W
lstm/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@Ж
lstm/zeros_1/packedPacklstm/strided_slice:output:0lstm/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:W
lstm/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Б
lstm/zeros_1Filllstm/zeros_1/packed:output:0lstm/zeros_1/Const:output:0*
T0*'
_output_shapes
:         @{
lstm/Read/ReadVariableOpReadVariableOp!lstm_read_readvariableop_resource*
_output_shapes
:		А*
dtype0e
lstm/IdentityIdentity lstm/Read/ReadVariableOp:value:0*
T0*
_output_shapes
:		А
lstm/Read_1/ReadVariableOpReadVariableOp#lstm_read_1_readvariableop_resource*
_output_shapes
:	@А*
dtype0i
lstm/Identity_1Identity"lstm/Read_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	@А{
lstm/Read_2/ReadVariableOpReadVariableOp#lstm_read_2_readvariableop_resource*
_output_shapes	
:А*
dtype0e
lstm/Identity_2Identity"lstm/Read_2/ReadVariableOp:value:0*
T0*
_output_shapes	
:Ат
lstm/PartitionedCallPartitionedCallnormalization/truediv:z:0lstm/zeros:output:0lstm/zeros_1:output:0lstm/Identity:output:0lstm/Identity_1:output:0lstm/Identity_2:output:0*
Tin

2*
Tout	
2*
_collective_manager_ids
 *f
_output_shapesT
R:         @:         @:         @:         @: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *(
f#R!
__inference_standard_lstm_22476Б
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	@э*
dtype0Н
dense/MatMulMatMullstm/PartitionedCall:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         э
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:э*
dtype0Й
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         эf
IdentityIdentitydense/BiasAdd:output:0^NoOp*
T0*(
_output_shapes
:         э╪
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^lstm/Read/ReadVariableOp^lstm/Read_1/ReadVariableOp^lstm/Read_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:'                           :	:	: : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp24
lstm/Read/ReadVariableOplstm/Read/ReadVariableOp28
lstm/Read_1/ReadVariableOplstm/Read_1/ReadVariableOp28
lstm/Read_2/ReadVariableOplstm/Read_2/ReadVariableOp:e a
=
_output_shapes+
):'                           
 
_user_specified_nameinputs:($
"
_output_shapes
:	:($
"
_output_shapes
:	
Г┬
у
9__inference___backward_gpu_lstm_with_fallback_23921_24097
placeholder
placeholder_1
placeholder_2
placeholder_3
placeholder_43
/gradients_expanddims_2_grad_shape_strided_slice)
%gradients_squeeze_grad_shape_cudnnrnn+
'gradients_squeeze_1_grad_shape_cudnnrnn/
+gradients_strided_slice_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9
5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_15
1gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h,
(gradients_expanddims_1_grad_shape_init_c-
)gradients_concat_1_grad_mod_concat_1_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_permA
=gradients_transpose_7_grad_invertpermutation_transpose_7_permA
=gradients_transpose_8_grad_invertpermutation_transpose_8_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim)
%gradients_concat_grad_mod_concat_axis
identity

identity_1

identity_2

identity_3

identity_4

identity_5И^
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:         @d
gradients/grad_ys_1Identityplaceholder_1*
T0*+
_output_shapes
:         @`
gradients/grad_ys_2Identityplaceholder_2*
T0*'
_output_shapes
:         @`
gradients/grad_ys_3Identityplaceholder_3*
T0*'
_output_shapes
:         @O
gradients/grad_ys_4Identityplaceholder_4*
T0*
_output_shapes
: А
!gradients/ExpandDims_2_grad/ShapeShape/gradients_expanddims_2_grad_shape_strided_slice*
T0*
_output_shapes
:к
#gradients/ExpandDims_2_grad/ReshapeReshapegradients/grad_ys_1:output:0*gradients/ExpandDims_2_grad/Shape:output:0*
T0*'
_output_shapes
:         @q
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
:д
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*+
_output_shapes
:         @u
gradients/Squeeze_1_grad/ShapeShape'gradients_squeeze_1_grad_shape_cudnnrnn*
T0*
_output_shapes
:и
 gradients/Squeeze_1_grad/ReshapeReshapegradients/grad_ys_3:output:0'gradients/Squeeze_1_grad/Shape:output:0*
T0*+
_output_shapes
:         @┼
gradients/AddNAddNgradients/grad_ys_0:output:0,gradients/ExpandDims_2_grad/Reshape:output:0*
N*
T0*&
_class
loc:@gradients/grad_ys_0*'
_output_shapes
:         @}
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:Ж
3gradients/strided_slice_grad/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
         {
1gradients/strided_slice_grad/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 
5gradients/strided_slice_grad/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:в
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0<gradients/strided_slice_grad/StridedSliceGrad/begin:output:0:gradients/strided_slice_grad/StridedSliceGrad/end:output:0>gradients/strided_slice_grad/StridedSliceGrad/strides:output:0gradients/AddN:sum:0*
Index0*
T0*4
_output_shapes"
 :                  @*
shrink_axis_maskc
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
:м
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_11gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnn'gradients_squeeze_1_grad_shape_cudnnrnn6gradients/strided_slice_grad/StridedSliceGrad:output:0'gradients/Squeeze_grad/Reshape:output:0)gradients/Squeeze_1_grad/Reshape:output:0gradients_zeros_like_cudnnrnn*
T0*j
_output_shapesX
V:                  	:         @:         @:АЦЦ
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:┘
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*4
_output_shapes"
 :                  	u
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:┼
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*'
_output_shapes
:         @y
!gradients/ExpandDims_1_grad/ShapeShape(gradients_expanddims_1_grad_shape_init_c*
T0*
_output_shapes
:╔
#gradients/ExpandDims_1_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_c_backprop:0*gradients/ExpandDims_1_grad/Shape:output:0*
T0*'
_output_shapes
:         @^
gradients/concat_1_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :Ъ
gradients/concat_1_grad/modFloorMod)gradients_concat_1_grad_mod_concat_1_axis%gradients/concat_1_grad/Rank:output:0*
T0*
_output_shapes
: h
gradients/concat_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:└j
gradients/concat_1_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:└j
gradients/concat_1_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:└j
gradients/concat_1_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:└j
gradients/concat_1_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:А j
gradients/concat_1_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:А j
gradients/concat_1_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB:А j
gradients/concat_1_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB:А i
gradients/concat_1_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB:@i
gradients/concat_1_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB:@j
 gradients/concat_1_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB:@j
 gradients/concat_1_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB:@j
 gradients/concat_1_grad/Shape_12Const*
_output_shapes
:*
dtype0*
valueB:@j
 gradients/concat_1_grad/Shape_13Const*
_output_shapes
:*
dtype0*
valueB:@j
 gradients/concat_1_grad/Shape_14Const*
_output_shapes
:*
dtype0*
valueB:@j
 gradients/concat_1_grad/Shape_15Const*
_output_shapes
:*
dtype0*
valueB:@°
$gradients/concat_1_grad/ConcatOffsetConcatOffsetgradients/concat_1_grad/mod:z:0&gradients/concat_1_grad/Shape:output:0(gradients/concat_1_grad/Shape_1:output:0(gradients/concat_1_grad/Shape_2:output:0(gradients/concat_1_grad/Shape_3:output:0(gradients/concat_1_grad/Shape_4:output:0(gradients/concat_1_grad/Shape_5:output:0(gradients/concat_1_grad/Shape_6:output:0(gradients/concat_1_grad/Shape_7:output:0(gradients/concat_1_grad/Shape_8:output:0(gradients/concat_1_grad/Shape_9:output:0)gradients/concat_1_grad/Shape_10:output:0)gradients/concat_1_grad/Shape_11:output:0)gradients/concat_1_grad/Shape_12:output:0)gradients/concat_1_grad/Shape_13:output:0)gradients/concat_1_grad/Shape_14:output:0)gradients/concat_1_grad/Shape_15:output:0*
N*t
_output_shapesb
`::::::::::::::::ь
gradients/concat_1_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:0&gradients/concat_1_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:└Ё
gradients/concat_1_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:1(gradients/concat_1_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:└Ё
gradients/concat_1_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:2(gradients/concat_1_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:└Ё
gradients/concat_1_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:3(gradients/concat_1_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:└Ё
gradients/concat_1_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:4(gradients/concat_1_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:А Ё
gradients/concat_1_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:5(gradients/concat_1_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:А Ё
gradients/concat_1_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:6(gradients/concat_1_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes	
:А Ё
gradients/concat_1_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:7(gradients/concat_1_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes	
:А я
gradients/concat_1_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:8(gradients/concat_1_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
:@я
gradients/concat_1_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:9(gradients/concat_1_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
:@Є
 gradients/concat_1_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:10)gradients/concat_1_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
:@Є
 gradients/concat_1_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:11)gradients/concat_1_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
:@Є
 gradients/concat_1_grad/Slice_12Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:12)gradients/concat_1_grad/Shape_12:output:0*
Index0*
T0*
_output_shapes
:@Є
 gradients/concat_1_grad/Slice_13Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:13)gradients/concat_1_grad/Shape_13:output:0*
Index0*
T0*
_output_shapes
:@Є
 gradients/concat_1_grad/Slice_14Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:14)gradients/concat_1_grad/Shape_14:output:0*
Index0*
T0*
_output_shapes
:@Є
 gradients/concat_1_grad/Slice_15Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:15)gradients/concat_1_grad/Shape_15:output:0*
Index0*
T0*
_output_shapes
:@m
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"@   	   б
gradients/Reshape_grad/ReshapeReshape&gradients/concat_1_grad/Slice:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes

:@	o
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"@   	   з
 gradients/Reshape_1_grad/ReshapeReshape(gradients/concat_1_grad/Slice_1:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes

:@	o
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"@   	   з
 gradients/Reshape_2_grad/ReshapeReshape(gradients/concat_1_grad/Slice_2:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes

:@	o
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"@   	   з
 gradients/Reshape_3_grad/ReshapeReshape(gradients/concat_1_grad/Slice_3:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes

:@	o
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"@   @   з
 gradients/Reshape_4_grad/ReshapeReshape(gradients/concat_1_grad/Slice_4:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:@@o
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"@   @   з
 gradients/Reshape_5_grad/ReshapeReshape(gradients/concat_1_grad/Slice_5:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:@@o
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"@   @   з
 gradients/Reshape_6_grad/ReshapeReshape(gradients/concat_1_grad/Slice_6:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:@@o
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"@   @   з
 gradients/Reshape_7_grad/ReshapeReshape(gradients/concat_1_grad/Slice_7:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes

:@@h
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:@г
 gradients/Reshape_8_grad/ReshapeReshape(gradients/concat_1_grad/Slice_8:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
:@h
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:@г
 gradients/Reshape_9_grad/ReshapeReshape(gradients/concat_1_grad/Slice_9:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
:@i
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:@ж
!gradients/Reshape_10_grad/ReshapeReshape)gradients/concat_1_grad/Slice_10:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
:@i
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:@ж
!gradients/Reshape_11_grad/ReshapeReshape)gradients/concat_1_grad/Slice_11:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
:@i
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:@ж
!gradients/Reshape_12_grad/ReshapeReshape)gradients/concat_1_grad/Slice_12:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
:@i
gradients/Reshape_13_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:@ж
!gradients/Reshape_13_grad/ReshapeReshape)gradients/concat_1_grad/Slice_13:output:0(gradients/Reshape_13_grad/Shape:output:0*
T0*
_output_shapes
:@i
gradients/Reshape_14_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:@ж
!gradients/Reshape_14_grad/ReshapeReshape)gradients/concat_1_grad/Slice_14:output:0(gradients/Reshape_14_grad/Shape:output:0*
T0*
_output_shapes
:@i
gradients/Reshape_15_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:@ж
!gradients/Reshape_15_grad/ReshapeReshape)gradients/concat_1_grad/Slice_15:output:0(gradients/Reshape_15_grad/Shape:output:0*
T0*
_output_shapes
:@Ь
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:╡
$gradients/transpose_1_grad/transpose	Transpose'gradients/Reshape_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes

:	@Ь
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:╖
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes

:	@Ь
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:╖
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes

:	@Ь
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:╖
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes

:	@Ь
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:╖
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@@Ь
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:╖
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@@Ь
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:╖
$gradients/transpose_7_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@@Ь
,gradients/transpose_8_grad/InvertPermutationInvertPermutation=gradients_transpose_8_grad_invertpermutation_transpose_8_perm*
_output_shapes
:╖
$gradients/transpose_8_grad/transpose	Transpose)gradients/Reshape_7_grad/Reshape:output:00gradients/transpose_8_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@@ч
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0*gradients/Reshape_13_grad/Reshape:output:0*gradients/Reshape_14_grad/Reshape:output:0*gradients/Reshape_15_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:Ап
gradients/split_grad/concatConcatV2(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0*
_output_shapes
:		А╡
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0(gradients/transpose_7_grad/transpose:y:0(gradients/transpose_8_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0*
_output_shapes
:	@А\
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :Т
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: f
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:Аh
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:А╩
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0*
N* 
_output_shapes
::╥
gradients/concat_grad/SliceSlice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:А╓
gradients/concat_grad/Slice_1Slice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:А{
IdentityIdentity&gradients/transpose_grad/transpose:y:0*
T0*4
_output_shapes"
 :                  	t

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0*
T0*'
_output_shapes
:         @v

Identity_2Identity,gradients/ExpandDims_1_grad/Reshape:output:0*
T0*'
_output_shapes
:         @f

Identity_3Identity$gradients/split_grad/concat:output:0*
T0*
_output_shapes
:		Аh

Identity_4Identity&gradients/split_1_grad/concat:output:0*
T0*
_output_shapes
:	@Аd

Identity_5Identity&gradients/concat_grad/Slice_1:output:0*
T0*
_output_shapes	
:А"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*(
_construction_contextkEagerRuntime*И
_input_shapesЎ
є:         @:         @:         @:         @: :         @:         @:         @:                  @::                  	:         @:         @:АЦ::         @:         @: ::::::::: : : : *=
api_implements+)lstm_5a52159d-dcb6-440d-9622-97ba434935b9*
api_preferred_deviceGPU*A
forward_function_name(&__forward_gpu_lstm_with_fallback_24096*
go_backwards( *

time_major( :- )
'
_output_shapes
:         @:1-
+
_output_shapes
:         @:-)
'
_output_shapes
:         @:-)
'
_output_shapes
:         @:

_output_shapes
: :-)
'
_output_shapes
:         @:1-
+
_output_shapes
:         @:1-
+
_output_shapes
:         @::6
4
_output_shapes"
 :                  @:	

_output_shapes
:::
6
4
_output_shapes"
 :                  	:1-
+
_output_shapes
:         @:1-
+
_output_shapes
:         @:"

_output_shapes

:АЦ: 

_output_shapes
::-)
'
_output_shapes
:         @:-)
'
_output_shapes
:         @:

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: "╡	L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*╫
serving_default├
i
normalization_inputR
%serving_default_normalization_input:0'                           :
dense1
StatefulPartitionedCall:0         эtensorflow/serving/predict:ФУ
█
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*	&call_and_return_all_conditional_losses

_default_save_signature
	optimizer

signatures"
_tf_keras_sequential
╙
	keras_api

_keep_axis
_reduce_axis
_reduce_axis_mask
_broadcast_shape
mean

adapt_mean
variance
adapt_variance
	count
_adapt_function"
_tf_keras_layer
┌
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_random_generator
cell

state_spec"
_tf_keras_rnn_layer
╗
	variables
 trainable_variables
!regularization_losses
"	keras_api
#__call__
*$&call_and_return_all_conditional_losses

%kernel
&bias"
_tf_keras_layer
X
0
1
2
'3
(4
)5
%6
&7"
trackable_list_wrapper
C
'0
(1
)2
%3
&4"
trackable_list_wrapper
 "
trackable_list_wrapper
╩
*non_trainable_variables

+layers
,metrics
-layer_regularization_losses
.layer_metrics
	variables
trainable_variables
regularization_losses
__call__

_default_save_signature
*	&call_and_return_all_conditional_losses
&	"call_and_return_conditional_losses"
_generic_user_object
▌
/trace_0
0trace_1
1trace_2
2trace_32Є
*__inference_sequential_layer_call_fn_21616
*__inference_sequential_layer_call_fn_22294
*__inference_sequential_layer_call_fn_22313
*__inference_sequential_layer_call_fn_22157┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z/trace_0z0trace_1z1trace_2z2trace_3
╔
3trace_0
4trace_1
5trace_2
6trace_32▐
E__inference_sequential_layer_call_and_return_conditional_losses_22755
E__inference_sequential_layer_call_and_return_conditional_losses_23197
E__inference_sequential_layer_call_and_return_conditional_losses_22180
E__inference_sequential_layer_call_and_return_conditional_losses_22203┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z3trace_0z4trace_1z5trace_2z6trace_3
У
7	capture_0
8	capture_1B╘
 __inference__wrapped_model_20250normalization_input"Ш
С▓Н
FullArgSpec
argsЪ 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z7	capture_0z8	capture_1
н
9iter

:beta_1

;beta_2
	<decay
=learning_rate%mm&mn'mo(mp)mq%vr&vs'vt(vu)vv"
	optimizer
,
>serving_default"
signature_map
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:	2mean
:	2variance
:	 2count
╪
?trace_02╗
__inference_adapt_step_22275Ъ
У▓П
FullArgSpec
argsЪ

jiterator
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z?trace_0
5
'0
(1
)2"
trackable_list_wrapper
5
'0
(1
)2"
trackable_list_wrapper
 "
trackable_list_wrapper
╣

@states
Anon_trainable_variables

Blayers
Cmetrics
Dlayer_regularization_losses
Elayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
┌
Ftrace_0
Gtrace_1
Htrace_2
Itrace_32я
$__inference_lstm_layer_call_fn_23208
$__inference_lstm_layer_call_fn_23219
$__inference_lstm_layer_call_fn_23230
$__inference_lstm_layer_call_fn_23241╘
╦▓╟
FullArgSpecB
args:Ъ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsЪ

 
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zFtrace_0zGtrace_1zHtrace_2zItrace_3
╞
Jtrace_0
Ktrace_1
Ltrace_2
Mtrace_32█
?__inference_lstm_layer_call_and_return_conditional_losses_23670
?__inference_lstm_layer_call_and_return_conditional_losses_24099
?__inference_lstm_layer_call_and_return_conditional_losses_24528
?__inference_lstm_layer_call_and_return_conditional_losses_24957╘
╦▓╟
FullArgSpecB
args:Ъ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsЪ

 
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zJtrace_0zKtrace_1zLtrace_2zMtrace_3
"
_generic_user_object
°
N	variables
Otrainable_variables
Pregularization_losses
Q	keras_api
R__call__
*S&call_and_return_all_conditional_losses
T_random_generator
U
state_size

'kernel
(recurrent_kernel
)bias"
_tf_keras_layer
 "
trackable_list_wrapper
.
%0
&1"
trackable_list_wrapper
.
%0
&1"
trackable_list_wrapper
 "
trackable_list_wrapper
н
Vnon_trainable_variables

Wlayers
Xmetrics
Ylayer_regularization_losses
Zlayer_metrics
	variables
 trainable_variables
!regularization_losses
#__call__
*$&call_and_return_all_conditional_losses
&$"call_and_return_conditional_losses"
_generic_user_object
щ
[trace_02╠
%__inference_dense_layer_call_fn_24966в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z[trace_0
Д
\trace_02ч
@__inference_dense_layer_call_and_return_conditional_losses_24976в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z\trace_0
:	@э2dense/kernel
:э2
dense/bias
(:&		А2lstm/lstm_cell/kernel
2:0	@А2lstm/lstm_cell/recurrent_kernel
": А2lstm/lstm_cell/bias
5
0
1
2"
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
.
]0
^1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
─
7	capture_0
8	capture_1BЕ
*__inference_sequential_layer_call_fn_21616normalization_input"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z7	capture_0z8	capture_1
╖
7	capture_0
8	capture_1B°
*__inference_sequential_layer_call_fn_22294inputs"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z7	capture_0z8	capture_1
╖
7	capture_0
8	capture_1B°
*__inference_sequential_layer_call_fn_22313inputs"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z7	capture_0z8	capture_1
─
7	capture_0
8	capture_1BЕ
*__inference_sequential_layer_call_fn_22157normalization_input"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z7	capture_0z8	capture_1
╥
7	capture_0
8	capture_1BУ
E__inference_sequential_layer_call_and_return_conditional_losses_22755inputs"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z7	capture_0z8	capture_1
╥
7	capture_0
8	capture_1BУ
E__inference_sequential_layer_call_and_return_conditional_losses_23197inputs"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z7	capture_0z8	capture_1
▀
7	capture_0
8	capture_1Bа
E__inference_sequential_layer_call_and_return_conditional_losses_22180normalization_input"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z7	capture_0z8	capture_1
▀
7	capture_0
8	capture_1Bа
E__inference_sequential_layer_call_and_return_conditional_losses_22203normalization_input"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z7	capture_0z8	capture_1
J
Constjtf.TrackableConstant
!J	
Const_1jtf.TrackableConstant
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
Т
7	capture_0
8	capture_1B╙
#__inference_signature_wrapper_22230normalization_input"Ф
Н▓Й
FullArgSpec
argsЪ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z7	capture_0z8	capture_1
╩B╟
__inference_adapt_step_22275iterator"Ъ
У▓П
FullArgSpec
argsЪ

jiterator
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
МBЙ
$__inference_lstm_layer_call_fn_23208inputs/0"╘
╦▓╟
FullArgSpecB
args:Ъ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsЪ

 
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
МBЙ
$__inference_lstm_layer_call_fn_23219inputs/0"╘
╦▓╟
FullArgSpecB
args:Ъ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsЪ

 
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
КBЗ
$__inference_lstm_layer_call_fn_23230inputs"╘
╦▓╟
FullArgSpecB
args:Ъ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsЪ

 
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
КBЗ
$__inference_lstm_layer_call_fn_23241inputs"╘
╦▓╟
FullArgSpecB
args:Ъ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsЪ

 
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
зBд
?__inference_lstm_layer_call_and_return_conditional_losses_23670inputs/0"╘
╦▓╟
FullArgSpecB
args:Ъ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsЪ

 
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
зBд
?__inference_lstm_layer_call_and_return_conditional_losses_24099inputs/0"╘
╦▓╟
FullArgSpecB
args:Ъ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsЪ

 
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
еBв
?__inference_lstm_layer_call_and_return_conditional_losses_24528inputs"╘
╦▓╟
FullArgSpecB
args:Ъ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsЪ

 
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
еBв
?__inference_lstm_layer_call_and_return_conditional_losses_24957inputs"╘
╦▓╟
FullArgSpecB
args:Ъ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsЪ

 
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
5
'0
(1
)2"
trackable_list_wrapper
5
'0
(1
)2"
trackable_list_wrapper
 "
trackable_list_wrapper
н
_non_trainable_variables

`layers
ametrics
blayer_regularization_losses
clayer_metrics
N	variables
Otrainable_variables
Pregularization_losses
R__call__
*S&call_and_return_all_conditional_losses
&S"call_and_return_conditional_losses"
_generic_user_object
├2└╜
┤▓░
FullArgSpec3
args+Ъ(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
├2└╜
┤▓░
FullArgSpec3
args+Ъ(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
┘B╓
%__inference_dense_layer_call_fn_24966inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЇBё
@__inference_dense_layer_call_and_return_conditional_losses_24976inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
N
d	variables
e	keras_api
	ftotal
	gcount"
_tf_keras_metric
^
h	variables
i	keras_api
	jtotal
	kcount
l
_fn_kwargs"
_tf_keras_metric
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
f0
g1"
trackable_list_wrapper
-
d	variables"
_generic_user_object
:  (2total
:  (2count
.
j0
k1"
trackable_list_wrapper
-
h	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
$:"	@э2Adam/dense/kernel/m
:э2Adam/dense/bias/m
-:+		А2Adam/lstm/lstm_cell/kernel/m
7:5	@А2&Adam/lstm/lstm_cell/recurrent_kernel/m
':%А2Adam/lstm/lstm_cell/bias/m
$:"	@э2Adam/dense/kernel/v
:э2Adam/dense/bias/v
-:+		А2Adam/lstm/lstm_cell/kernel/v
7:5	@А2&Adam/lstm/lstm_cell/recurrent_kernel/v
':%А2Adam/lstm/lstm_cell/bias/v▓
 __inference__wrapped_model_20250Н78'()%&RвO
HвE
CК@
normalization_input'                           
к ".к+
)
dense К
dense         эs
__inference_adapt_step_22275SHвE
>в;
9Т6"в
К         ┌	IteratorSpec 
к "
 б
@__inference_dense_layer_call_and_return_conditional_losses_24976]%&/в,
%в"
 К
inputs         @
к "&в#
К
0         э
Ъ y
%__inference_dense_layer_call_fn_24966P%&/в,
%в"
 К
inputs         @
к "К         э└
?__inference_lstm_layer_call_and_return_conditional_losses_23670}'()OвL
EвB
4Ъ1
/К,
inputs/0                  	

 
p 

 
к "%в"
К
0         @
Ъ └
?__inference_lstm_layer_call_and_return_conditional_losses_24099}'()OвL
EвB
4Ъ1
/К,
inputs/0                  	

 
p

 
к "%в"
К
0         @
Ъ ╣
?__inference_lstm_layer_call_and_return_conditional_losses_24528v'()HвE
>в;
-К*
inputs                  	

 
p 

 
к "%в"
К
0         @
Ъ ╣
?__inference_lstm_layer_call_and_return_conditional_losses_24957v'()HвE
>в;
-К*
inputs                  	

 
p

 
к "%в"
К
0         @
Ъ Ш
$__inference_lstm_layer_call_fn_23208p'()OвL
EвB
4Ъ1
/К,
inputs/0                  	

 
p 

 
к "К         @Ш
$__inference_lstm_layer_call_fn_23219p'()OвL
EвB
4Ъ1
/К,
inputs/0                  	

 
p

 
к "К         @С
$__inference_lstm_layer_call_fn_23230i'()HвE
>в;
-К*
inputs                  	

 
p 

 
к "К         @С
$__inference_lstm_layer_call_fn_23241i'()HвE
>в;
-К*
inputs                  	

 
p

 
к "К         @╫
E__inference_sequential_layer_call_and_return_conditional_losses_22180Н78'()%&ZвW
PвM
CК@
normalization_input'                           
p 

 
к "&в#
К
0         э
Ъ ╫
E__inference_sequential_layer_call_and_return_conditional_losses_22203Н78'()%&ZвW
PвM
CК@
normalization_input'                           
p

 
к "&в#
К
0         э
Ъ ╩
E__inference_sequential_layer_call_and_return_conditional_losses_22755А78'()%&MвJ
Cв@
6К3
inputs'                           
p 

 
к "&в#
К
0         э
Ъ ╩
E__inference_sequential_layer_call_and_return_conditional_losses_23197А78'()%&MвJ
Cв@
6К3
inputs'                           
p

 
к "&в#
К
0         э
Ъ п
*__inference_sequential_layer_call_fn_21616А78'()%&ZвW
PвM
CК@
normalization_input'                           
p 

 
к "К         эп
*__inference_sequential_layer_call_fn_22157А78'()%&ZвW
PвM
CК@
normalization_input'                           
p

 
к "К         эб
*__inference_sequential_layer_call_fn_22294s78'()%&MвJ
Cв@
6К3
inputs'                           
p 

 
к "К         эб
*__inference_sequential_layer_call_fn_22313s78'()%&MвJ
Cв@
6К3
inputs'                           
p

 
к "К         э╠
#__inference_signature_wrapper_22230д78'()%&iвf
в 
_к\
Z
normalization_inputCК@
normalization_input'                           ".к+
)
dense К
dense         э