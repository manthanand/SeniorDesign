��
��
D
AddV2
x"T
y"T
z"T"
Ttype:
2	��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
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
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �
?
Mul
x"T
y"T
z"T"
Ttype:
2	�

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
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
�
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
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
�
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
�
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type/
output_handle���element_dtype"
element_dtypetype"

shape_typetype:
2	
�
TensorListReserve
element_shape"
shape_type
num_elements(
handle���element_dtype"
element_dtypetype"

shape_typetype:
2	
�
TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsint���������
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �
�
While

input2T
output2T"
T
list(type)("
condfunc"
bodyfunc" 
output_shapeslist(shape)
 "
parallel_iterationsint
�"serve*2.10.02v2.10.0-rc3-6-g359c3cdfc5f8��
�
 Adam/lstm_66/lstm_cell_66/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*1
shared_name" Adam/lstm_66/lstm_cell_66/bias/v
�
4Adam/lstm_66/lstm_cell_66/bias/v/Read/ReadVariableOpReadVariableOp Adam/lstm_66/lstm_cell_66/bias/v*
_output_shapes	
:�*
dtype0
�
,Adam/lstm_66/lstm_cell_66/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	d�*=
shared_name.,Adam/lstm_66/lstm_cell_66/recurrent_kernel/v
�
@Adam/lstm_66/lstm_cell_66/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp,Adam/lstm_66/lstm_cell_66/recurrent_kernel/v*
_output_shapes
:	d�*
dtype0
�
"Adam/lstm_66/lstm_cell_66/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*3
shared_name$"Adam/lstm_66/lstm_cell_66/kernel/v
�
6Adam/lstm_66/lstm_cell_66/kernel/v/Read/ReadVariableOpReadVariableOp"Adam/lstm_66/lstm_cell_66/kernel/v*
_output_shapes
:	�*
dtype0
�
Adam/dense_200/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_200/bias/v
{
)Adam/dense_200/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_200/bias/v*
_output_shapes
:*
dtype0
�
Adam/dense_200/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*(
shared_nameAdam/dense_200/kernel/v
�
+Adam/dense_200/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_200/kernel/v*
_output_shapes

:2*
dtype0
�
Adam/dense_199/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*&
shared_nameAdam/dense_199/bias/v
{
)Adam/dense_199/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_199/bias/v*
_output_shapes
:2*
dtype0
�
Adam/dense_199/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:22*(
shared_nameAdam/dense_199/kernel/v
�
+Adam/dense_199/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_199/kernel/v*
_output_shapes

:22*
dtype0
�
Adam/dense_198/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*&
shared_nameAdam/dense_198/bias/v
{
)Adam/dense_198/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_198/bias/v*
_output_shapes
:2*
dtype0
�
Adam/dense_198/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d2*(
shared_nameAdam/dense_198/kernel/v
�
+Adam/dense_198/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_198/kernel/v*
_output_shapes

:d2*
dtype0
�
 Adam/lstm_66/lstm_cell_66/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*1
shared_name" Adam/lstm_66/lstm_cell_66/bias/m
�
4Adam/lstm_66/lstm_cell_66/bias/m/Read/ReadVariableOpReadVariableOp Adam/lstm_66/lstm_cell_66/bias/m*
_output_shapes	
:�*
dtype0
�
,Adam/lstm_66/lstm_cell_66/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	d�*=
shared_name.,Adam/lstm_66/lstm_cell_66/recurrent_kernel/m
�
@Adam/lstm_66/lstm_cell_66/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp,Adam/lstm_66/lstm_cell_66/recurrent_kernel/m*
_output_shapes
:	d�*
dtype0
�
"Adam/lstm_66/lstm_cell_66/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*3
shared_name$"Adam/lstm_66/lstm_cell_66/kernel/m
�
6Adam/lstm_66/lstm_cell_66/kernel/m/Read/ReadVariableOpReadVariableOp"Adam/lstm_66/lstm_cell_66/kernel/m*
_output_shapes
:	�*
dtype0
�
Adam/dense_200/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_200/bias/m
{
)Adam/dense_200/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_200/bias/m*
_output_shapes
:*
dtype0
�
Adam/dense_200/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*(
shared_nameAdam/dense_200/kernel/m
�
+Adam/dense_200/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_200/kernel/m*
_output_shapes

:2*
dtype0
�
Adam/dense_199/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*&
shared_nameAdam/dense_199/bias/m
{
)Adam/dense_199/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_199/bias/m*
_output_shapes
:2*
dtype0
�
Adam/dense_199/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:22*(
shared_nameAdam/dense_199/kernel/m
�
+Adam/dense_199/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_199/kernel/m*
_output_shapes

:22*
dtype0
�
Adam/dense_198/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*&
shared_nameAdam/dense_198/bias/m
{
)Adam/dense_198/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_198/bias/m*
_output_shapes
:2*
dtype0
�
Adam/dense_198/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d2*(
shared_nameAdam/dense_198/kernel/m
�
+Adam/dense_198/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_198/kernel/m*
_output_shapes

:d2*
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
�
lstm_66/lstm_cell_66/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�**
shared_namelstm_66/lstm_cell_66/bias
�
-lstm_66/lstm_cell_66/bias/Read/ReadVariableOpReadVariableOplstm_66/lstm_cell_66/bias*
_output_shapes	
:�*
dtype0
�
%lstm_66/lstm_cell_66/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	d�*6
shared_name'%lstm_66/lstm_cell_66/recurrent_kernel
�
9lstm_66/lstm_cell_66/recurrent_kernel/Read/ReadVariableOpReadVariableOp%lstm_66/lstm_cell_66/recurrent_kernel*
_output_shapes
:	d�*
dtype0
�
lstm_66/lstm_cell_66/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*,
shared_namelstm_66/lstm_cell_66/kernel
�
/lstm_66/lstm_cell_66/kernel/Read/ReadVariableOpReadVariableOplstm_66/lstm_cell_66/kernel*
_output_shapes
:	�*
dtype0
t
dense_200/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_200/bias
m
"dense_200/bias/Read/ReadVariableOpReadVariableOpdense_200/bias*
_output_shapes
:*
dtype0
|
dense_200/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*!
shared_namedense_200/kernel
u
$dense_200/kernel/Read/ReadVariableOpReadVariableOpdense_200/kernel*
_output_shapes

:2*
dtype0
t
dense_199/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*
shared_namedense_199/bias
m
"dense_199/bias/Read/ReadVariableOpReadVariableOpdense_199/bias*
_output_shapes
:2*
dtype0
|
dense_199/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:22*!
shared_namedense_199/kernel
u
$dense_199/kernel/Read/ReadVariableOpReadVariableOpdense_199/kernel*
_output_shapes

:22*
dtype0
t
dense_198/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*
shared_namedense_198/bias
m
"dense_198/bias/Read/ReadVariableOpReadVariableOpdense_198/bias*
_output_shapes
:2*
dtype0
|
dense_198/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d2*!
shared_namedense_198/kernel
u
$dense_198/kernel/Read/ReadVariableOpReadVariableOpdense_198/kernel*
_output_shapes

:d2*
dtype0
�
serving_default_lstm_66_inputPlaceholder*+
_output_shapes
:���������*
dtype0* 
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_lstm_66_inputlstm_66/lstm_cell_66/kernel%lstm_66/lstm_cell_66/recurrent_kernellstm_66/lstm_cell_66/biasdense_198/kerneldense_198/biasdense_199/kerneldense_199/biasdense_200/kerneldense_200/bias*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*+
_read_only_resource_inputs
		*-
config_proto

CPU

GPU 2J 8� *0
f+R)
'__inference_signature_wrapper_437463810

NoOpNoOp
�C
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�C
value�CB�C B�C
�
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
	variables
trainable_variables
regularization_losses
	keras_api
	__call__
*
&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_random_generator
cell

state_spec*
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias*
�
	variables
 trainable_variables
!regularization_losses
"	keras_api
#__call__
*$&call_and_return_all_conditional_losses

%kernel
&bias*
�
'	variables
(trainable_variables
)regularization_losses
*	keras_api
+__call__
*,&call_and_return_all_conditional_losses

-kernel
.bias*
C
/0
01
12
3
4
%5
&6
-7
.8*
C
/0
01
12
3
4
%5
&6
-7
.8*
* 
�
2non_trainable_variables

3layers
4metrics
5layer_regularization_losses
6layer_metrics
	variables
trainable_variables
regularization_losses
	__call__
_default_save_signature
*
&call_and_return_all_conditional_losses
&
"call_and_return_conditional_losses*
6
7trace_0
8trace_1
9trace_2
:trace_3* 
6
;trace_0
<trace_1
=trace_2
>trace_3* 
* 
�
?iter

@beta_1

Abeta_2
	Bdecay
Clearning_ratem�m�%m�&m�-m�.m�/m�0m�1m�v�v�%v�&v�-v�.v�/v�0v�1v�*

Dserving_default* 

/0
01
12*

/0
01
12*
* 
�

Estates
Fnon_trainable_variables

Glayers
Hmetrics
Ilayer_regularization_losses
Jlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
Ktrace_0
Ltrace_1
Mtrace_2
Ntrace_3* 
6
Otrace_0
Ptrace_1
Qtrace_2
Rtrace_3* 
* 
�
S	variables
Ttrainable_variables
Uregularization_losses
V	keras_api
W__call__
*X&call_and_return_all_conditional_losses
Y_random_generator
Z
state_size

/kernel
0recurrent_kernel
1bias*
* 

0
1*

0
1*
* 
�
[non_trainable_variables

\layers
]metrics
^layer_regularization_losses
_layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

`trace_0* 

atrace_0* 
`Z
VARIABLE_VALUEdense_198/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_198/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

%0
&1*

%0
&1*
* 
�
bnon_trainable_variables

clayers
dmetrics
elayer_regularization_losses
flayer_metrics
	variables
 trainable_variables
!regularization_losses
#__call__
*$&call_and_return_all_conditional_losses
&$"call_and_return_conditional_losses*

gtrace_0* 

htrace_0* 
`Z
VARIABLE_VALUEdense_199/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_199/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

-0
.1*

-0
.1*
* 
�
inon_trainable_variables

jlayers
kmetrics
llayer_regularization_losses
mlayer_metrics
'	variables
(trainable_variables
)regularization_losses
+__call__
*,&call_and_return_all_conditional_losses
&,"call_and_return_conditional_losses*

ntrace_0* 

otrace_0* 
`Z
VARIABLE_VALUEdense_200/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_200/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUElstm_66/lstm_cell_66/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUE%lstm_66/lstm_cell_66/recurrent_kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUElstm_66/lstm_cell_66/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
0
1
2
3*

p0
q1*
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
* 
* 
* 

0*
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
/0
01
12*

/0
01
12*
* 
�
rnon_trainable_variables

slayers
tmetrics
ulayer_regularization_losses
vlayer_metrics
S	variables
Ttrainable_variables
Uregularization_losses
W__call__
*X&call_and_return_all_conditional_losses
&X"call_and_return_conditional_losses*

wtrace_0
xtrace_1* 

ytrace_0
ztrace_1* 
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
* 
8
{	variables
|	keras_api
	}total
	~count*
L
	variables
�	keras_api

�total

�count
�
_fn_kwargs*
* 
* 
* 
* 
* 
* 
* 
* 
* 

}0
~1*

{	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1*

	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
�}
VARIABLE_VALUEAdam/dense_198/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_198/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�}
VARIABLE_VALUEAdam/dense_199/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_199/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�}
VARIABLE_VALUEAdam/dense_200/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_200/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE"Adam/lstm_66/lstm_cell_66/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE,Adam/lstm_66/lstm_cell_66/recurrent_kernel/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE Adam/lstm_66/lstm_cell_66/bias/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�}
VARIABLE_VALUEAdam/dense_198/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_198/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�}
VARIABLE_VALUEAdam/dense_199/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_199/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�}
VARIABLE_VALUEAdam/dense_200/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_200/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE"Adam/lstm_66/lstm_cell_66/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE,Adam/lstm_66/lstm_cell_66/recurrent_kernel/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE Adam/lstm_66/lstm_cell_66/bias/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$dense_198/kernel/Read/ReadVariableOp"dense_198/bias/Read/ReadVariableOp$dense_199/kernel/Read/ReadVariableOp"dense_199/bias/Read/ReadVariableOp$dense_200/kernel/Read/ReadVariableOp"dense_200/bias/Read/ReadVariableOp/lstm_66/lstm_cell_66/kernel/Read/ReadVariableOp9lstm_66/lstm_cell_66/recurrent_kernel/Read/ReadVariableOp-lstm_66/lstm_cell_66/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp+Adam/dense_198/kernel/m/Read/ReadVariableOp)Adam/dense_198/bias/m/Read/ReadVariableOp+Adam/dense_199/kernel/m/Read/ReadVariableOp)Adam/dense_199/bias/m/Read/ReadVariableOp+Adam/dense_200/kernel/m/Read/ReadVariableOp)Adam/dense_200/bias/m/Read/ReadVariableOp6Adam/lstm_66/lstm_cell_66/kernel/m/Read/ReadVariableOp@Adam/lstm_66/lstm_cell_66/recurrent_kernel/m/Read/ReadVariableOp4Adam/lstm_66/lstm_cell_66/bias/m/Read/ReadVariableOp+Adam/dense_198/kernel/v/Read/ReadVariableOp)Adam/dense_198/bias/v/Read/ReadVariableOp+Adam/dense_199/kernel/v/Read/ReadVariableOp)Adam/dense_199/bias/v/Read/ReadVariableOp+Adam/dense_200/kernel/v/Read/ReadVariableOp)Adam/dense_200/bias/v/Read/ReadVariableOp6Adam/lstm_66/lstm_cell_66/kernel/v/Read/ReadVariableOp@Adam/lstm_66/lstm_cell_66/recurrent_kernel/v/Read/ReadVariableOp4Adam/lstm_66/lstm_cell_66/bias/v/Read/ReadVariableOpConst*1
Tin*
(2&	*
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
GPU 2J 8� *+
f&R$
"__inference__traced_save_437465098
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_198/kerneldense_198/biasdense_199/kerneldense_199/biasdense_200/kerneldense_200/biaslstm_66/lstm_cell_66/kernel%lstm_66/lstm_cell_66/recurrent_kernellstm_66/lstm_cell_66/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotal_1count_1totalcountAdam/dense_198/kernel/mAdam/dense_198/bias/mAdam/dense_199/kernel/mAdam/dense_199/bias/mAdam/dense_200/kernel/mAdam/dense_200/bias/m"Adam/lstm_66/lstm_cell_66/kernel/m,Adam/lstm_66/lstm_cell_66/recurrent_kernel/m Adam/lstm_66/lstm_cell_66/bias/mAdam/dense_198/kernel/vAdam/dense_198/bias/vAdam/dense_199/kernel/vAdam/dense_199/bias/vAdam/dense_200/kernel/vAdam/dense_200/bias/v"Adam/lstm_66/lstm_cell_66/kernel/v,Adam/lstm_66/lstm_cell_66/recurrent_kernel/v Adam/lstm_66/lstm_cell_66/bias/v*0
Tin)
'2%*
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
GPU 2J 8� *.
f)R'
%__inference__traced_restore_437465216��
�L
�
"__inference__traced_save_437465098
file_prefix/
+savev2_dense_198_kernel_read_readvariableop-
)savev2_dense_198_bias_read_readvariableop/
+savev2_dense_199_kernel_read_readvariableop-
)savev2_dense_199_bias_read_readvariableop/
+savev2_dense_200_kernel_read_readvariableop-
)savev2_dense_200_bias_read_readvariableop:
6savev2_lstm_66_lstm_cell_66_kernel_read_readvariableopD
@savev2_lstm_66_lstm_cell_66_recurrent_kernel_read_readvariableop8
4savev2_lstm_66_lstm_cell_66_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop6
2savev2_adam_dense_198_kernel_m_read_readvariableop4
0savev2_adam_dense_198_bias_m_read_readvariableop6
2savev2_adam_dense_199_kernel_m_read_readvariableop4
0savev2_adam_dense_199_bias_m_read_readvariableop6
2savev2_adam_dense_200_kernel_m_read_readvariableop4
0savev2_adam_dense_200_bias_m_read_readvariableopA
=savev2_adam_lstm_66_lstm_cell_66_kernel_m_read_readvariableopK
Gsavev2_adam_lstm_66_lstm_cell_66_recurrent_kernel_m_read_readvariableop?
;savev2_adam_lstm_66_lstm_cell_66_bias_m_read_readvariableop6
2savev2_adam_dense_198_kernel_v_read_readvariableop4
0savev2_adam_dense_198_bias_v_read_readvariableop6
2savev2_adam_dense_199_kernel_v_read_readvariableop4
0savev2_adam_dense_199_bias_v_read_readvariableop6
2savev2_adam_dense_200_kernel_v_read_readvariableop4
0savev2_adam_dense_200_bias_v_read_readvariableopA
=savev2_adam_lstm_66_lstm_cell_66_kernel_v_read_readvariableopK
Gsavev2_adam_lstm_66_lstm_cell_66_recurrent_kernel_v_read_readvariableop?
;savev2_adam_lstm_66_lstm_cell_66_bias_v_read_readvariableop
savev2_const

identity_1��MergeV2Checkpointsw
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
_temp/part�
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
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:%*
dtype0*�
value�B�%B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:%*
dtype0*]
valueTBR%B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_dense_198_kernel_read_readvariableop)savev2_dense_198_bias_read_readvariableop+savev2_dense_199_kernel_read_readvariableop)savev2_dense_199_bias_read_readvariableop+savev2_dense_200_kernel_read_readvariableop)savev2_dense_200_bias_read_readvariableop6savev2_lstm_66_lstm_cell_66_kernel_read_readvariableop@savev2_lstm_66_lstm_cell_66_recurrent_kernel_read_readvariableop4savev2_lstm_66_lstm_cell_66_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop2savev2_adam_dense_198_kernel_m_read_readvariableop0savev2_adam_dense_198_bias_m_read_readvariableop2savev2_adam_dense_199_kernel_m_read_readvariableop0savev2_adam_dense_199_bias_m_read_readvariableop2savev2_adam_dense_200_kernel_m_read_readvariableop0savev2_adam_dense_200_bias_m_read_readvariableop=savev2_adam_lstm_66_lstm_cell_66_kernel_m_read_readvariableopGsavev2_adam_lstm_66_lstm_cell_66_recurrent_kernel_m_read_readvariableop;savev2_adam_lstm_66_lstm_cell_66_bias_m_read_readvariableop2savev2_adam_dense_198_kernel_v_read_readvariableop0savev2_adam_dense_198_bias_v_read_readvariableop2savev2_adam_dense_199_kernel_v_read_readvariableop0savev2_adam_dense_199_bias_v_read_readvariableop2savev2_adam_dense_200_kernel_v_read_readvariableop0savev2_adam_dense_200_bias_v_read_readvariableop=savev2_adam_lstm_66_lstm_cell_66_kernel_v_read_readvariableopGsavev2_adam_lstm_66_lstm_cell_66_recurrent_kernel_v_read_readvariableop;savev2_adam_lstm_66_lstm_cell_66_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *3
dtypes)
'2%	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
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

identity_1Identity_1:output:0*�
_input_shapes�
�: :d2:2:22:2:2::	�:	d�:�: : : : : : : : : :d2:2:22:2:2::	�:	d�:�:d2:2:22:2:2::	�:	d�:�: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:d2: 

_output_shapes
:2:$ 

_output_shapes

:22: 

_output_shapes
:2:$ 

_output_shapes

:2: 

_output_shapes
::%!

_output_shapes
:	�:%!

_output_shapes
:	d�:!	

_output_shapes	
:�:
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
: :

_output_shapes
: :$ 

_output_shapes

:d2: 

_output_shapes
:2:$ 

_output_shapes

:22: 

_output_shapes
:2:$ 

_output_shapes

:2: 

_output_shapes
::%!

_output_shapes
:	�:%!

_output_shapes
:	d�:!

_output_shapes	
:�:$ 

_output_shapes

:d2: 

_output_shapes
:2:$ 

_output_shapes

:22: 

_output_shapes
:2:$  

_output_shapes

:2: !

_output_shapes
::%"!

_output_shapes
:	�:%#!

_output_shapes
:	d�:!$

_output_shapes	
:�:%

_output_shapes
: 
�K
�
F__inference_lstm_66_layer_call_and_return_conditional_losses_437464520
inputs_0>
+lstm_cell_66_matmul_readvariableop_resource:	�@
-lstm_cell_66_matmul_1_readvariableop_resource:	d�;
,lstm_cell_66_biasadd_readvariableop_resource:	�
identity��#lstm_cell_66/BiasAdd/ReadVariableOp�"lstm_cell_66/MatMul/ReadVariableOp�$lstm_cell_66/MatMul_1/ReadVariableOp�while=
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
valueB:�
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
value	B :ds
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
:���������dR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :dw
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
:���������dc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :������������������D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
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
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
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
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
"lstm_cell_66/MatMul/ReadVariableOpReadVariableOp+lstm_cell_66_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
lstm_cell_66/MatMulMatMulstrided_slice_2:output:0*lstm_cell_66/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
$lstm_cell_66/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_66_matmul_1_readvariableop_resource*
_output_shapes
:	d�*
dtype0�
lstm_cell_66/MatMul_1MatMulzeros:output:0,lstm_cell_66/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
lstm_cell_66/addAddV2lstm_cell_66/MatMul:product:0lstm_cell_66/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
#lstm_cell_66/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_66_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
lstm_cell_66/BiasAddBiasAddlstm_cell_66/add:z:0+lstm_cell_66/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������^
lstm_cell_66/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_cell_66/splitSplit%lstm_cell_66/split/split_dim:output:0lstm_cell_66/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������d:���������d:���������d:���������d*
	num_splitn
lstm_cell_66/SigmoidSigmoidlstm_cell_66/split:output:0*
T0*'
_output_shapes
:���������dp
lstm_cell_66/Sigmoid_1Sigmoidlstm_cell_66/split:output:1*
T0*'
_output_shapes
:���������dw
lstm_cell_66/mulMullstm_cell_66/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������dh
lstm_cell_66/ReluRelulstm_cell_66/split:output:2*
T0*'
_output_shapes
:���������d�
lstm_cell_66/mul_1Mullstm_cell_66/Sigmoid:y:0lstm_cell_66/Relu:activations:0*
T0*'
_output_shapes
:���������d{
lstm_cell_66/add_1AddV2lstm_cell_66/mul:z:0lstm_cell_66/mul_1:z:0*
T0*'
_output_shapes
:���������dp
lstm_cell_66/Sigmoid_2Sigmoidlstm_cell_66/split:output:3*
T0*'
_output_shapes
:���������de
lstm_cell_66/Relu_1Relulstm_cell_66/add_1:z:0*
T0*'
_output_shapes
:���������d�
lstm_cell_66/mul_2Mullstm_cell_66/Sigmoid_2:y:0!lstm_cell_66/Relu_1:activations:0*
T0*'
_output_shapes
:���������dn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
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
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_66_matmul_readvariableop_resource-lstm_cell_66_matmul_1_readvariableop_resource,lstm_cell_66_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������d:���������d: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( * 
bodyR
while_body_437464435* 
condR
while_cond_437464434*K
output_shapes:
8: : : : :���������d:���������d: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������d*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������d*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:���������d[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:���������d�
NoOpNoOp$^lstm_cell_66/BiasAdd/ReadVariableOp#^lstm_cell_66/MatMul/ReadVariableOp%^lstm_cell_66/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 2J
#lstm_cell_66/BiasAdd/ReadVariableOp#lstm_cell_66/BiasAdd/ReadVariableOp2H
"lstm_cell_66/MatMul/ReadVariableOp"lstm_cell_66/MatMul/ReadVariableOp2L
$lstm_cell_66/MatMul_1/ReadVariableOp$lstm_cell_66/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :������������������
"
_user_specified_name
inputs/0
�$
�
while_body_437462932
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_01
while_lstm_cell_66_437462956_0:	�1
while_lstm_cell_66_437462958_0:	d�-
while_lstm_cell_66_437462960_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor/
while_lstm_cell_66_437462956:	�/
while_lstm_cell_66_437462958:	d�+
while_lstm_cell_66_437462960:	���*while/lstm_cell_66/StatefulPartitionedCall�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
*while/lstm_cell_66/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_66_437462956_0while_lstm_cell_66_437462958_0while_lstm_cell_66_437462960_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:���������d:���������d:���������d*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_lstm_cell_66_layer_call_and_return_conditional_losses_437462917r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : �
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:03while/lstm_cell_66/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_4Identity3while/lstm_cell_66/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:���������d�
while/Identity_5Identity3while/lstm_cell_66/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:���������dy

while/NoOpNoOp+^while/lstm_cell_66/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0">
while_lstm_cell_66_437462956while_lstm_cell_66_437462956_0">
while_lstm_cell_66_437462958while_lstm_cell_66_437462958_0">
while_lstm_cell_66_437462960while_lstm_cell_66_437462960_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������d:���������d: : : : : 2X
*while/lstm_cell_66/StatefulPartitionedCall*while/lstm_cell_66/StatefulPartitionedCall: 
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
:���������d:-)
'
_output_shapes
:���������d:

_output_shapes
: :

_output_shapes
: 
�
�
while_cond_437463537
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_17
3while_while_cond_437463537___redundant_placeholder07
3while_while_cond_437463537___redundant_placeholder17
3while_while_cond_437463537___redundant_placeholder27
3while_while_cond_437463537___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
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
@: : : : :���������d:���������d: ::::: 
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
:���������d:-)
'
_output_shapes
:���������d:

_output_shapes
: :

_output_shapes
:
�
�
while_cond_437462931
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_17
3while_while_cond_437462931___redundant_placeholder07
3while_while_cond_437462931___redundant_placeholder17
3while_while_cond_437462931___redundant_placeholder27
3while_while_cond_437462931___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
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
@: : : : :���������d:���������d: ::::: 
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
:���������d:-)
'
_output_shapes
:���������d:

_output_shapes
: :

_output_shapes
:
�B
�

lstm_66_while_body_437464081,
(lstm_66_while_lstm_66_while_loop_counter2
.lstm_66_while_lstm_66_while_maximum_iterations
lstm_66_while_placeholder
lstm_66_while_placeholder_1
lstm_66_while_placeholder_2
lstm_66_while_placeholder_3+
'lstm_66_while_lstm_66_strided_slice_1_0g
clstm_66_while_tensorarrayv2read_tensorlistgetitem_lstm_66_tensorarrayunstack_tensorlistfromtensor_0N
;lstm_66_while_lstm_cell_66_matmul_readvariableop_resource_0:	�P
=lstm_66_while_lstm_cell_66_matmul_1_readvariableop_resource_0:	d�K
<lstm_66_while_lstm_cell_66_biasadd_readvariableop_resource_0:	�
lstm_66_while_identity
lstm_66_while_identity_1
lstm_66_while_identity_2
lstm_66_while_identity_3
lstm_66_while_identity_4
lstm_66_while_identity_5)
%lstm_66_while_lstm_66_strided_slice_1e
alstm_66_while_tensorarrayv2read_tensorlistgetitem_lstm_66_tensorarrayunstack_tensorlistfromtensorL
9lstm_66_while_lstm_cell_66_matmul_readvariableop_resource:	�N
;lstm_66_while_lstm_cell_66_matmul_1_readvariableop_resource:	d�I
:lstm_66_while_lstm_cell_66_biasadd_readvariableop_resource:	���1lstm_66/while/lstm_cell_66/BiasAdd/ReadVariableOp�0lstm_66/while/lstm_cell_66/MatMul/ReadVariableOp�2lstm_66/while/lstm_cell_66/MatMul_1/ReadVariableOp�
?lstm_66/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
1lstm_66/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_66_while_tensorarrayv2read_tensorlistgetitem_lstm_66_tensorarrayunstack_tensorlistfromtensor_0lstm_66_while_placeholderHlstm_66/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
0lstm_66/while/lstm_cell_66/MatMul/ReadVariableOpReadVariableOp;lstm_66_while_lstm_cell_66_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
!lstm_66/while/lstm_cell_66/MatMulMatMul8lstm_66/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_66/while/lstm_cell_66/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
2lstm_66/while/lstm_cell_66/MatMul_1/ReadVariableOpReadVariableOp=lstm_66_while_lstm_cell_66_matmul_1_readvariableop_resource_0*
_output_shapes
:	d�*
dtype0�
#lstm_66/while/lstm_cell_66/MatMul_1MatMullstm_66_while_placeholder_2:lstm_66/while/lstm_cell_66/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
lstm_66/while/lstm_cell_66/addAddV2+lstm_66/while/lstm_cell_66/MatMul:product:0-lstm_66/while/lstm_cell_66/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
1lstm_66/while/lstm_cell_66/BiasAdd/ReadVariableOpReadVariableOp<lstm_66_while_lstm_cell_66_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype0�
"lstm_66/while/lstm_cell_66/BiasAddBiasAdd"lstm_66/while/lstm_cell_66/add:z:09lstm_66/while/lstm_cell_66/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������l
*lstm_66/while/lstm_cell_66/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
 lstm_66/while/lstm_cell_66/splitSplit3lstm_66/while/lstm_cell_66/split/split_dim:output:0+lstm_66/while/lstm_cell_66/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������d:���������d:���������d:���������d*
	num_split�
"lstm_66/while/lstm_cell_66/SigmoidSigmoid)lstm_66/while/lstm_cell_66/split:output:0*
T0*'
_output_shapes
:���������d�
$lstm_66/while/lstm_cell_66/Sigmoid_1Sigmoid)lstm_66/while/lstm_cell_66/split:output:1*
T0*'
_output_shapes
:���������d�
lstm_66/while/lstm_cell_66/mulMul(lstm_66/while/lstm_cell_66/Sigmoid_1:y:0lstm_66_while_placeholder_3*
T0*'
_output_shapes
:���������d�
lstm_66/while/lstm_cell_66/ReluRelu)lstm_66/while/lstm_cell_66/split:output:2*
T0*'
_output_shapes
:���������d�
 lstm_66/while/lstm_cell_66/mul_1Mul&lstm_66/while/lstm_cell_66/Sigmoid:y:0-lstm_66/while/lstm_cell_66/Relu:activations:0*
T0*'
_output_shapes
:���������d�
 lstm_66/while/lstm_cell_66/add_1AddV2"lstm_66/while/lstm_cell_66/mul:z:0$lstm_66/while/lstm_cell_66/mul_1:z:0*
T0*'
_output_shapes
:���������d�
$lstm_66/while/lstm_cell_66/Sigmoid_2Sigmoid)lstm_66/while/lstm_cell_66/split:output:3*
T0*'
_output_shapes
:���������d�
!lstm_66/while/lstm_cell_66/Relu_1Relu$lstm_66/while/lstm_cell_66/add_1:z:0*
T0*'
_output_shapes
:���������d�
 lstm_66/while/lstm_cell_66/mul_2Mul(lstm_66/while/lstm_cell_66/Sigmoid_2:y:0/lstm_66/while/lstm_cell_66/Relu_1:activations:0*
T0*'
_output_shapes
:���������dz
8lstm_66/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : �
2lstm_66/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_66_while_placeholder_1Alstm_66/while/TensorArrayV2Write/TensorListSetItem/index:output:0$lstm_66/while/lstm_cell_66/mul_2:z:0*
_output_shapes
: *
element_dtype0:���U
lstm_66/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :t
lstm_66/while/addAddV2lstm_66_while_placeholderlstm_66/while/add/y:output:0*
T0*
_output_shapes
: W
lstm_66/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_66/while/add_1AddV2(lstm_66_while_lstm_66_while_loop_counterlstm_66/while/add_1/y:output:0*
T0*
_output_shapes
: q
lstm_66/while/IdentityIdentitylstm_66/while/add_1:z:0^lstm_66/while/NoOp*
T0*
_output_shapes
: �
lstm_66/while/Identity_1Identity.lstm_66_while_lstm_66_while_maximum_iterations^lstm_66/while/NoOp*
T0*
_output_shapes
: q
lstm_66/while/Identity_2Identitylstm_66/while/add:z:0^lstm_66/while/NoOp*
T0*
_output_shapes
: �
lstm_66/while/Identity_3IdentityBlstm_66/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_66/while/NoOp*
T0*
_output_shapes
: �
lstm_66/while/Identity_4Identity$lstm_66/while/lstm_cell_66/mul_2:z:0^lstm_66/while/NoOp*
T0*'
_output_shapes
:���������d�
lstm_66/while/Identity_5Identity$lstm_66/while/lstm_cell_66/add_1:z:0^lstm_66/while/NoOp*
T0*'
_output_shapes
:���������d�
lstm_66/while/NoOpNoOp2^lstm_66/while/lstm_cell_66/BiasAdd/ReadVariableOp1^lstm_66/while/lstm_cell_66/MatMul/ReadVariableOp3^lstm_66/while/lstm_cell_66/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "9
lstm_66_while_identitylstm_66/while/Identity:output:0"=
lstm_66_while_identity_1!lstm_66/while/Identity_1:output:0"=
lstm_66_while_identity_2!lstm_66/while/Identity_2:output:0"=
lstm_66_while_identity_3!lstm_66/while/Identity_3:output:0"=
lstm_66_while_identity_4!lstm_66/while/Identity_4:output:0"=
lstm_66_while_identity_5!lstm_66/while/Identity_5:output:0"P
%lstm_66_while_lstm_66_strided_slice_1'lstm_66_while_lstm_66_strided_slice_1_0"z
:lstm_66_while_lstm_cell_66_biasadd_readvariableop_resource<lstm_66_while_lstm_cell_66_biasadd_readvariableop_resource_0"|
;lstm_66_while_lstm_cell_66_matmul_1_readvariableop_resource=lstm_66_while_lstm_cell_66_matmul_1_readvariableop_resource_0"x
9lstm_66_while_lstm_cell_66_matmul_readvariableop_resource;lstm_66_while_lstm_cell_66_matmul_readvariableop_resource_0"�
alstm_66_while_tensorarrayv2read_tensorlistgetitem_lstm_66_tensorarrayunstack_tensorlistfromtensorclstm_66_while_tensorarrayv2read_tensorlistgetitem_lstm_66_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������d:���������d: : : : : 2f
1lstm_66/while/lstm_cell_66/BiasAdd/ReadVariableOp1lstm_66/while/lstm_cell_66/BiasAdd/ReadVariableOp2d
0lstm_66/while/lstm_cell_66/MatMul/ReadVariableOp0lstm_66/while/lstm_cell_66/MatMul/ReadVariableOp2h
2lstm_66/while/lstm_cell_66/MatMul_1/ReadVariableOp2lstm_66/while/lstm_cell_66/MatMul_1/ReadVariableOp: 
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
:���������d:-)
'
_output_shapes
:���������d:

_output_shapes
: :

_output_shapes
: 
�
�
K__inference_lstm_cell_66_layer_call_and_return_conditional_losses_437464935

inputs
states_0
states_11
matmul_readvariableop_resource:	�3
 matmul_1_readvariableop_resource:	d�.
biasadd_readvariableop_resource:	�
identity

identity_1

identity_2��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������y
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	d�*
dtype0p
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������e
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:���������d:���������d:���������d:���������d*
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:���������dV
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:���������dU
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:���������dN
ReluRelusplit:output:2*
T0*'
_output_shapes
:���������d_
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:���������dT
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:���������dV
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:���������dK
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:���������dc
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:���������dX
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:���������dZ

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:���������dZ

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:���������d�
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������:���������d:���������d: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:QM
'
_output_shapes
:���������d
"
_user_specified_name
states/0:QM
'
_output_shapes
:���������d
"
_user_specified_name
states/1
�
�
0__inference_lstm_cell_66_layer_call_fn_437464886

inputs
states_0
states_1
unknown:	�
	unknown_0:	d�
	unknown_1:	�
identity

identity_1

identity_2��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:���������d:���������d:���������d*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_lstm_cell_66_layer_call_and_return_conditional_losses_437462917o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������dq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:���������dq

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:���������d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������:���������d:���������d: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:QM
'
_output_shapes
:���������d
"
_user_specified_name
states/0:QM
'
_output_shapes
:���������d
"
_user_specified_name
states/1
�	
�
H__inference_dense_200_layer_call_and_return_conditional_losses_437463407

inputs0
matmul_readvariableop_resource:2-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:2*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������2: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������2
 
_user_specified_nameinputs
�
�	
$__inference__wrapped_model_437462850
lstm_66_inputT
Asequential_66_lstm_66_lstm_cell_66_matmul_readvariableop_resource:	�V
Csequential_66_lstm_66_lstm_cell_66_matmul_1_readvariableop_resource:	d�Q
Bsequential_66_lstm_66_lstm_cell_66_biasadd_readvariableop_resource:	�H
6sequential_66_dense_198_matmul_readvariableop_resource:d2E
7sequential_66_dense_198_biasadd_readvariableop_resource:2H
6sequential_66_dense_199_matmul_readvariableop_resource:22E
7sequential_66_dense_199_biasadd_readvariableop_resource:2H
6sequential_66_dense_200_matmul_readvariableop_resource:2E
7sequential_66_dense_200_biasadd_readvariableop_resource:
identity��.sequential_66/dense_198/BiasAdd/ReadVariableOp�-sequential_66/dense_198/MatMul/ReadVariableOp�.sequential_66/dense_199/BiasAdd/ReadVariableOp�-sequential_66/dense_199/MatMul/ReadVariableOp�.sequential_66/dense_200/BiasAdd/ReadVariableOp�-sequential_66/dense_200/MatMul/ReadVariableOp�9sequential_66/lstm_66/lstm_cell_66/BiasAdd/ReadVariableOp�8sequential_66/lstm_66/lstm_cell_66/MatMul/ReadVariableOp�:sequential_66/lstm_66/lstm_cell_66/MatMul_1/ReadVariableOp�sequential_66/lstm_66/whileX
sequential_66/lstm_66/ShapeShapelstm_66_input*
T0*
_output_shapes
:s
)sequential_66/lstm_66/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+sequential_66/lstm_66/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+sequential_66/lstm_66/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
#sequential_66/lstm_66/strided_sliceStridedSlice$sequential_66/lstm_66/Shape:output:02sequential_66/lstm_66/strided_slice/stack:output:04sequential_66/lstm_66/strided_slice/stack_1:output:04sequential_66/lstm_66/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
$sequential_66/lstm_66/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d�
"sequential_66/lstm_66/zeros/packedPack,sequential_66/lstm_66/strided_slice:output:0-sequential_66/lstm_66/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:f
!sequential_66/lstm_66/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
sequential_66/lstm_66/zerosFill+sequential_66/lstm_66/zeros/packed:output:0*sequential_66/lstm_66/zeros/Const:output:0*
T0*'
_output_shapes
:���������dh
&sequential_66/lstm_66/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d�
$sequential_66/lstm_66/zeros_1/packedPack,sequential_66/lstm_66/strided_slice:output:0/sequential_66/lstm_66/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:h
#sequential_66/lstm_66/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
sequential_66/lstm_66/zeros_1Fill-sequential_66/lstm_66/zeros_1/packed:output:0,sequential_66/lstm_66/zeros_1/Const:output:0*
T0*'
_output_shapes
:���������dy
$sequential_66/lstm_66/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
sequential_66/lstm_66/transpose	Transposelstm_66_input-sequential_66/lstm_66/transpose/perm:output:0*
T0*+
_output_shapes
:���������p
sequential_66/lstm_66/Shape_1Shape#sequential_66/lstm_66/transpose:y:0*
T0*
_output_shapes
:u
+sequential_66/lstm_66/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-sequential_66/lstm_66/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-sequential_66/lstm_66/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
%sequential_66/lstm_66/strided_slice_1StridedSlice&sequential_66/lstm_66/Shape_1:output:04sequential_66/lstm_66/strided_slice_1/stack:output:06sequential_66/lstm_66/strided_slice_1/stack_1:output:06sequential_66/lstm_66/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask|
1sequential_66/lstm_66/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
#sequential_66/lstm_66/TensorArrayV2TensorListReserve:sequential_66/lstm_66/TensorArrayV2/element_shape:output:0.sequential_66/lstm_66/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
Ksequential_66/lstm_66/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
=sequential_66/lstm_66/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor#sequential_66/lstm_66/transpose:y:0Tsequential_66/lstm_66/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���u
+sequential_66/lstm_66/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-sequential_66/lstm_66/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-sequential_66/lstm_66/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
%sequential_66/lstm_66/strided_slice_2StridedSlice#sequential_66/lstm_66/transpose:y:04sequential_66/lstm_66/strided_slice_2/stack:output:06sequential_66/lstm_66/strided_slice_2/stack_1:output:06sequential_66/lstm_66/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
8sequential_66/lstm_66/lstm_cell_66/MatMul/ReadVariableOpReadVariableOpAsequential_66_lstm_66_lstm_cell_66_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
)sequential_66/lstm_66/lstm_cell_66/MatMulMatMul.sequential_66/lstm_66/strided_slice_2:output:0@sequential_66/lstm_66/lstm_cell_66/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
:sequential_66/lstm_66/lstm_cell_66/MatMul_1/ReadVariableOpReadVariableOpCsequential_66_lstm_66_lstm_cell_66_matmul_1_readvariableop_resource*
_output_shapes
:	d�*
dtype0�
+sequential_66/lstm_66/lstm_cell_66/MatMul_1MatMul$sequential_66/lstm_66/zeros:output:0Bsequential_66/lstm_66/lstm_cell_66/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
&sequential_66/lstm_66/lstm_cell_66/addAddV23sequential_66/lstm_66/lstm_cell_66/MatMul:product:05sequential_66/lstm_66/lstm_cell_66/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
9sequential_66/lstm_66/lstm_cell_66/BiasAdd/ReadVariableOpReadVariableOpBsequential_66_lstm_66_lstm_cell_66_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
*sequential_66/lstm_66/lstm_cell_66/BiasAddBiasAdd*sequential_66/lstm_66/lstm_cell_66/add:z:0Asequential_66/lstm_66/lstm_cell_66/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������t
2sequential_66/lstm_66/lstm_cell_66/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
(sequential_66/lstm_66/lstm_cell_66/splitSplit;sequential_66/lstm_66/lstm_cell_66/split/split_dim:output:03sequential_66/lstm_66/lstm_cell_66/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������d:���������d:���������d:���������d*
	num_split�
*sequential_66/lstm_66/lstm_cell_66/SigmoidSigmoid1sequential_66/lstm_66/lstm_cell_66/split:output:0*
T0*'
_output_shapes
:���������d�
,sequential_66/lstm_66/lstm_cell_66/Sigmoid_1Sigmoid1sequential_66/lstm_66/lstm_cell_66/split:output:1*
T0*'
_output_shapes
:���������d�
&sequential_66/lstm_66/lstm_cell_66/mulMul0sequential_66/lstm_66/lstm_cell_66/Sigmoid_1:y:0&sequential_66/lstm_66/zeros_1:output:0*
T0*'
_output_shapes
:���������d�
'sequential_66/lstm_66/lstm_cell_66/ReluRelu1sequential_66/lstm_66/lstm_cell_66/split:output:2*
T0*'
_output_shapes
:���������d�
(sequential_66/lstm_66/lstm_cell_66/mul_1Mul.sequential_66/lstm_66/lstm_cell_66/Sigmoid:y:05sequential_66/lstm_66/lstm_cell_66/Relu:activations:0*
T0*'
_output_shapes
:���������d�
(sequential_66/lstm_66/lstm_cell_66/add_1AddV2*sequential_66/lstm_66/lstm_cell_66/mul:z:0,sequential_66/lstm_66/lstm_cell_66/mul_1:z:0*
T0*'
_output_shapes
:���������d�
,sequential_66/lstm_66/lstm_cell_66/Sigmoid_2Sigmoid1sequential_66/lstm_66/lstm_cell_66/split:output:3*
T0*'
_output_shapes
:���������d�
)sequential_66/lstm_66/lstm_cell_66/Relu_1Relu,sequential_66/lstm_66/lstm_cell_66/add_1:z:0*
T0*'
_output_shapes
:���������d�
(sequential_66/lstm_66/lstm_cell_66/mul_2Mul0sequential_66/lstm_66/lstm_cell_66/Sigmoid_2:y:07sequential_66/lstm_66/lstm_cell_66/Relu_1:activations:0*
T0*'
_output_shapes
:���������d�
3sequential_66/lstm_66/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   t
2sequential_66/lstm_66/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :�
%sequential_66/lstm_66/TensorArrayV2_1TensorListReserve<sequential_66/lstm_66/TensorArrayV2_1/element_shape:output:0;sequential_66/lstm_66/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���\
sequential_66/lstm_66/timeConst*
_output_shapes
: *
dtype0*
value	B : y
.sequential_66/lstm_66/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������j
(sequential_66/lstm_66/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
sequential_66/lstm_66/whileWhile1sequential_66/lstm_66/while/loop_counter:output:07sequential_66/lstm_66/while/maximum_iterations:output:0#sequential_66/lstm_66/time:output:0.sequential_66/lstm_66/TensorArrayV2_1:handle:0$sequential_66/lstm_66/zeros:output:0&sequential_66/lstm_66/zeros_1:output:0.sequential_66/lstm_66/strided_slice_1:output:0Msequential_66/lstm_66/TensorArrayUnstack/TensorListFromTensor:output_handle:0Asequential_66_lstm_66_lstm_cell_66_matmul_readvariableop_resourceCsequential_66_lstm_66_lstm_cell_66_matmul_1_readvariableop_resourceBsequential_66_lstm_66_lstm_cell_66_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������d:���������d: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *6
body.R,
*sequential_66_lstm_66_while_body_437462745*6
cond.R,
*sequential_66_lstm_66_while_cond_437462744*K
output_shapes:
8: : : : :���������d:���������d: : : : : *
parallel_iterations �
Fsequential_66/lstm_66/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   �
8sequential_66/lstm_66/TensorArrayV2Stack/TensorListStackTensorListStack$sequential_66/lstm_66/while:output:3Osequential_66/lstm_66/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������d*
element_dtype0*
num_elements~
+sequential_66/lstm_66/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������w
-sequential_66/lstm_66/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: w
-sequential_66/lstm_66/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
%sequential_66/lstm_66/strided_slice_3StridedSliceAsequential_66/lstm_66/TensorArrayV2Stack/TensorListStack:tensor:04sequential_66/lstm_66/strided_slice_3/stack:output:06sequential_66/lstm_66/strided_slice_3/stack_1:output:06sequential_66/lstm_66/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������d*
shrink_axis_mask{
&sequential_66/lstm_66/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
!sequential_66/lstm_66/transpose_1	TransposeAsequential_66/lstm_66/TensorArrayV2Stack/TensorListStack:tensor:0/sequential_66/lstm_66/transpose_1/perm:output:0*
T0*+
_output_shapes
:���������dq
sequential_66/lstm_66/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    �
-sequential_66/dense_198/MatMul/ReadVariableOpReadVariableOp6sequential_66_dense_198_matmul_readvariableop_resource*
_output_shapes

:d2*
dtype0�
sequential_66/dense_198/MatMulMatMul.sequential_66/lstm_66/strided_slice_3:output:05sequential_66/dense_198/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2�
.sequential_66/dense_198/BiasAdd/ReadVariableOpReadVariableOp7sequential_66_dense_198_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0�
sequential_66/dense_198/BiasAddBiasAdd(sequential_66/dense_198/MatMul:product:06sequential_66/dense_198/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2�
sequential_66/dense_198/ReluRelu(sequential_66/dense_198/BiasAdd:output:0*
T0*'
_output_shapes
:���������2�
-sequential_66/dense_199/MatMul/ReadVariableOpReadVariableOp6sequential_66_dense_199_matmul_readvariableop_resource*
_output_shapes

:22*
dtype0�
sequential_66/dense_199/MatMulMatMul*sequential_66/dense_198/Relu:activations:05sequential_66/dense_199/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2�
.sequential_66/dense_199/BiasAdd/ReadVariableOpReadVariableOp7sequential_66_dense_199_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0�
sequential_66/dense_199/BiasAddBiasAdd(sequential_66/dense_199/MatMul:product:06sequential_66/dense_199/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2�
sequential_66/dense_199/ReluRelu(sequential_66/dense_199/BiasAdd:output:0*
T0*'
_output_shapes
:���������2�
-sequential_66/dense_200/MatMul/ReadVariableOpReadVariableOp6sequential_66_dense_200_matmul_readvariableop_resource*
_output_shapes

:2*
dtype0�
sequential_66/dense_200/MatMulMatMul*sequential_66/dense_199/Relu:activations:05sequential_66/dense_200/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
.sequential_66/dense_200/BiasAdd/ReadVariableOpReadVariableOp7sequential_66_dense_200_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
sequential_66/dense_200/BiasAddBiasAdd(sequential_66/dense_200/MatMul:product:06sequential_66/dense_200/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������w
IdentityIdentity(sequential_66/dense_200/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp/^sequential_66/dense_198/BiasAdd/ReadVariableOp.^sequential_66/dense_198/MatMul/ReadVariableOp/^sequential_66/dense_199/BiasAdd/ReadVariableOp.^sequential_66/dense_199/MatMul/ReadVariableOp/^sequential_66/dense_200/BiasAdd/ReadVariableOp.^sequential_66/dense_200/MatMul/ReadVariableOp:^sequential_66/lstm_66/lstm_cell_66/BiasAdd/ReadVariableOp9^sequential_66/lstm_66/lstm_cell_66/MatMul/ReadVariableOp;^sequential_66/lstm_66/lstm_cell_66/MatMul_1/ReadVariableOp^sequential_66/lstm_66/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):���������: : : : : : : : : 2`
.sequential_66/dense_198/BiasAdd/ReadVariableOp.sequential_66/dense_198/BiasAdd/ReadVariableOp2^
-sequential_66/dense_198/MatMul/ReadVariableOp-sequential_66/dense_198/MatMul/ReadVariableOp2`
.sequential_66/dense_199/BiasAdd/ReadVariableOp.sequential_66/dense_199/BiasAdd/ReadVariableOp2^
-sequential_66/dense_199/MatMul/ReadVariableOp-sequential_66/dense_199/MatMul/ReadVariableOp2`
.sequential_66/dense_200/BiasAdd/ReadVariableOp.sequential_66/dense_200/BiasAdd/ReadVariableOp2^
-sequential_66/dense_200/MatMul/ReadVariableOp-sequential_66/dense_200/MatMul/ReadVariableOp2v
9sequential_66/lstm_66/lstm_cell_66/BiasAdd/ReadVariableOp9sequential_66/lstm_66/lstm_cell_66/BiasAdd/ReadVariableOp2t
8sequential_66/lstm_66/lstm_cell_66/MatMul/ReadVariableOp8sequential_66/lstm_66/lstm_cell_66/MatMul/ReadVariableOp2x
:sequential_66/lstm_66/lstm_cell_66/MatMul_1/ReadVariableOp:sequential_66/lstm_66/lstm_cell_66/MatMul_1/ReadVariableOp2:
sequential_66/lstm_66/whilesequential_66/lstm_66/while:Z V
+
_output_shapes
:���������
'
_user_specified_namelstm_66_input
�
�
0__inference_lstm_cell_66_layer_call_fn_437464903

inputs
states_0
states_1
unknown:	�
	unknown_0:	d�
	unknown_1:	�
identity

identity_1

identity_2��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:���������d:���������d:���������d*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_lstm_cell_66_layer_call_and_return_conditional_losses_437463065o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������dq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:���������dq

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:���������d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������:���������d:���������d: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:QM
'
_output_shapes
:���������d
"
_user_specified_name
states/0:QM
'
_output_shapes
:���������d
"
_user_specified_name
states/1
�
�
-__inference_dense_198_layer_call_fn_437464819

inputs
unknown:d2
	unknown_0:2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_dense_198_layer_call_and_return_conditional_losses_437463374o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������d: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
L__inference_sequential_66_layer_call_and_return_conditional_losses_437463779
lstm_66_input$
lstm_66_437463756:	�$
lstm_66_437463758:	d� 
lstm_66_437463760:	�%
dense_198_437463763:d2!
dense_198_437463765:2%
dense_199_437463768:22!
dense_199_437463770:2%
dense_200_437463773:2!
dense_200_437463775:
identity��!dense_198/StatefulPartitionedCall�!dense_199/StatefulPartitionedCall�!dense_200/StatefulPartitionedCall�lstm_66/StatefulPartitionedCall�
lstm_66/StatefulPartitionedCallStatefulPartitionedCalllstm_66_inputlstm_66_437463756lstm_66_437463758lstm_66_437463760*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_lstm_66_layer_call_and_return_conditional_losses_437463623�
!dense_198/StatefulPartitionedCallStatefulPartitionedCall(lstm_66/StatefulPartitionedCall:output:0dense_198_437463763dense_198_437463765*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_dense_198_layer_call_and_return_conditional_losses_437463374�
!dense_199/StatefulPartitionedCallStatefulPartitionedCall*dense_198/StatefulPartitionedCall:output:0dense_199_437463768dense_199_437463770*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_dense_199_layer_call_and_return_conditional_losses_437463391�
!dense_200/StatefulPartitionedCallStatefulPartitionedCall*dense_199/StatefulPartitionedCall:output:0dense_200_437463773dense_200_437463775*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_dense_200_layer_call_and_return_conditional_losses_437463407y
IdentityIdentity*dense_200/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp"^dense_198/StatefulPartitionedCall"^dense_199/StatefulPartitionedCall"^dense_200/StatefulPartitionedCall ^lstm_66/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):���������: : : : : : : : : 2F
!dense_198/StatefulPartitionedCall!dense_198/StatefulPartitionedCall2F
!dense_199/StatefulPartitionedCall!dense_199/StatefulPartitionedCall2F
!dense_200/StatefulPartitionedCall!dense_200/StatefulPartitionedCall2B
lstm_66/StatefulPartitionedCalllstm_66/StatefulPartitionedCall:Z V
+
_output_shapes
:���������
'
_user_specified_namelstm_66_input
�9
�
while_body_437463538
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_66_matmul_readvariableop_resource_0:	�H
5while_lstm_cell_66_matmul_1_readvariableop_resource_0:	d�C
4while_lstm_cell_66_biasadd_readvariableop_resource_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_66_matmul_readvariableop_resource:	�F
3while_lstm_cell_66_matmul_1_readvariableop_resource:	d�A
2while_lstm_cell_66_biasadd_readvariableop_resource:	���)while/lstm_cell_66/BiasAdd/ReadVariableOp�(while/lstm_cell_66/MatMul/ReadVariableOp�*while/lstm_cell_66/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
(while/lstm_cell_66/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_66_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
while/lstm_cell_66/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_66/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
*while/lstm_cell_66/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_66_matmul_1_readvariableop_resource_0*
_output_shapes
:	d�*
dtype0�
while/lstm_cell_66/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_66/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
while/lstm_cell_66/addAddV2#while/lstm_cell_66/MatMul:product:0%while/lstm_cell_66/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
)while/lstm_cell_66/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_66_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype0�
while/lstm_cell_66/BiasAddBiasAddwhile/lstm_cell_66/add:z:01while/lstm_cell_66/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������d
"while/lstm_cell_66/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/lstm_cell_66/splitSplit+while/lstm_cell_66/split/split_dim:output:0#while/lstm_cell_66/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������d:���������d:���������d:���������d*
	num_splitz
while/lstm_cell_66/SigmoidSigmoid!while/lstm_cell_66/split:output:0*
T0*'
_output_shapes
:���������d|
while/lstm_cell_66/Sigmoid_1Sigmoid!while/lstm_cell_66/split:output:1*
T0*'
_output_shapes
:���������d�
while/lstm_cell_66/mulMul while/lstm_cell_66/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������dt
while/lstm_cell_66/ReluRelu!while/lstm_cell_66/split:output:2*
T0*'
_output_shapes
:���������d�
while/lstm_cell_66/mul_1Mulwhile/lstm_cell_66/Sigmoid:y:0%while/lstm_cell_66/Relu:activations:0*
T0*'
_output_shapes
:���������d�
while/lstm_cell_66/add_1AddV2while/lstm_cell_66/mul:z:0while/lstm_cell_66/mul_1:z:0*
T0*'
_output_shapes
:���������d|
while/lstm_cell_66/Sigmoid_2Sigmoid!while/lstm_cell_66/split:output:3*
T0*'
_output_shapes
:���������dq
while/lstm_cell_66/Relu_1Reluwhile/lstm_cell_66/add_1:z:0*
T0*'
_output_shapes
:���������d�
while/lstm_cell_66/mul_2Mul while/lstm_cell_66/Sigmoid_2:y:0'while/lstm_cell_66/Relu_1:activations:0*
T0*'
_output_shapes
:���������dr
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : �
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell_66/mul_2:z:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: y
while/Identity_4Identitywhile/lstm_cell_66/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:���������dy
while/Identity_5Identitywhile/lstm_cell_66/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:���������d�

while/NoOpNoOp*^while/lstm_cell_66/BiasAdd/ReadVariableOp)^while/lstm_cell_66/MatMul/ReadVariableOp+^while/lstm_cell_66/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_66_biasadd_readvariableop_resource4while_lstm_cell_66_biasadd_readvariableop_resource_0"l
3while_lstm_cell_66_matmul_1_readvariableop_resource5while_lstm_cell_66_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_66_matmul_readvariableop_resource3while_lstm_cell_66_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������d:���������d: : : : : 2V
)while/lstm_cell_66/BiasAdd/ReadVariableOp)while/lstm_cell_66/BiasAdd/ReadVariableOp2T
(while/lstm_cell_66/MatMul/ReadVariableOp(while/lstm_cell_66/MatMul/ReadVariableOp2X
*while/lstm_cell_66/MatMul_1/ReadVariableOp*while/lstm_cell_66/MatMul_1/ReadVariableOp: 
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
:���������d:-)
'
_output_shapes
:���������d:

_output_shapes
: :

_output_shapes
: 
�K
�
F__inference_lstm_66_layer_call_and_return_conditional_losses_437464810

inputs>
+lstm_cell_66_matmul_readvariableop_resource:	�@
-lstm_cell_66_matmul_1_readvariableop_resource:	d�;
,lstm_cell_66_biasadd_readvariableop_resource:	�
identity��#lstm_cell_66/BiasAdd/ReadVariableOp�"lstm_cell_66/MatMul/ReadVariableOp�$lstm_cell_66/MatMul_1/ReadVariableOp�while;
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
valueB:�
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
value	B :ds
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
:���������dR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :dw
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
:���������dc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:���������D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
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
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
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
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
"lstm_cell_66/MatMul/ReadVariableOpReadVariableOp+lstm_cell_66_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
lstm_cell_66/MatMulMatMulstrided_slice_2:output:0*lstm_cell_66/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
$lstm_cell_66/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_66_matmul_1_readvariableop_resource*
_output_shapes
:	d�*
dtype0�
lstm_cell_66/MatMul_1MatMulzeros:output:0,lstm_cell_66/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
lstm_cell_66/addAddV2lstm_cell_66/MatMul:product:0lstm_cell_66/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
#lstm_cell_66/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_66_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
lstm_cell_66/BiasAddBiasAddlstm_cell_66/add:z:0+lstm_cell_66/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������^
lstm_cell_66/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_cell_66/splitSplit%lstm_cell_66/split/split_dim:output:0lstm_cell_66/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������d:���������d:���������d:���������d*
	num_splitn
lstm_cell_66/SigmoidSigmoidlstm_cell_66/split:output:0*
T0*'
_output_shapes
:���������dp
lstm_cell_66/Sigmoid_1Sigmoidlstm_cell_66/split:output:1*
T0*'
_output_shapes
:���������dw
lstm_cell_66/mulMullstm_cell_66/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������dh
lstm_cell_66/ReluRelulstm_cell_66/split:output:2*
T0*'
_output_shapes
:���������d�
lstm_cell_66/mul_1Mullstm_cell_66/Sigmoid:y:0lstm_cell_66/Relu:activations:0*
T0*'
_output_shapes
:���������d{
lstm_cell_66/add_1AddV2lstm_cell_66/mul:z:0lstm_cell_66/mul_1:z:0*
T0*'
_output_shapes
:���������dp
lstm_cell_66/Sigmoid_2Sigmoidlstm_cell_66/split:output:3*
T0*'
_output_shapes
:���������de
lstm_cell_66/Relu_1Relulstm_cell_66/add_1:z:0*
T0*'
_output_shapes
:���������d�
lstm_cell_66/mul_2Mullstm_cell_66/Sigmoid_2:y:0!lstm_cell_66/Relu_1:activations:0*
T0*'
_output_shapes
:���������dn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
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
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_66_matmul_readvariableop_resource-lstm_cell_66_matmul_1_readvariableop_resource,lstm_cell_66_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������d:���������d: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( * 
bodyR
while_body_437464725* 
condR
while_cond_437464724*K
output_shapes:
8: : : : :���������d:���������d: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������d*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������d*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:���������d[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:���������d�
NoOpNoOp$^lstm_cell_66/BiasAdd/ReadVariableOp#^lstm_cell_66/MatMul/ReadVariableOp%^lstm_cell_66/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������: : : 2J
#lstm_cell_66/BiasAdd/ReadVariableOp#lstm_cell_66/BiasAdd/ReadVariableOp2H
"lstm_cell_66/MatMul/ReadVariableOp"lstm_cell_66/MatMul/ReadVariableOp2L
$lstm_cell_66/MatMul_1/ReadVariableOp$lstm_cell_66/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
H__inference_dense_199_layer_call_and_return_conditional_losses_437464850

inputs0
matmul_readvariableop_resource:22-
biasadd_readvariableop_resource:2
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:22*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������2a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������2w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������2: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������2
 
_user_specified_nameinputs
�
�
-__inference_dense_199_layer_call_fn_437464839

inputs
unknown:22
	unknown_0:2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_dense_199_layer_call_and_return_conditional_losses_437463391o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������2: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������2
 
_user_specified_nameinputs
�$
�
while_body_437463125
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_01
while_lstm_cell_66_437463149_0:	�1
while_lstm_cell_66_437463151_0:	d�-
while_lstm_cell_66_437463153_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor/
while_lstm_cell_66_437463149:	�/
while_lstm_cell_66_437463151:	d�+
while_lstm_cell_66_437463153:	���*while/lstm_cell_66/StatefulPartitionedCall�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
*while/lstm_cell_66/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_66_437463149_0while_lstm_cell_66_437463151_0while_lstm_cell_66_437463153_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:���������d:���������d:���������d*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_lstm_cell_66_layer_call_and_return_conditional_losses_437463065r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : �
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:03while/lstm_cell_66/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_4Identity3while/lstm_cell_66/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:���������d�
while/Identity_5Identity3while/lstm_cell_66/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:���������dy

while/NoOpNoOp+^while/lstm_cell_66/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0">
while_lstm_cell_66_437463149while_lstm_cell_66_437463149_0">
while_lstm_cell_66_437463151while_lstm_cell_66_437463151_0">
while_lstm_cell_66_437463153while_lstm_cell_66_437463153_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������d:���������d: : : : : 2X
*while/lstm_cell_66/StatefulPartitionedCall*while/lstm_cell_66/StatefulPartitionedCall: 
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
:���������d:-)
'
_output_shapes
:���������d:

_output_shapes
: :

_output_shapes
: 
�

�
lstm_66_while_cond_437463915,
(lstm_66_while_lstm_66_while_loop_counter2
.lstm_66_while_lstm_66_while_maximum_iterations
lstm_66_while_placeholder
lstm_66_while_placeholder_1
lstm_66_while_placeholder_2
lstm_66_while_placeholder_3.
*lstm_66_while_less_lstm_66_strided_slice_1G
Clstm_66_while_lstm_66_while_cond_437463915___redundant_placeholder0G
Clstm_66_while_lstm_66_while_cond_437463915___redundant_placeholder1G
Clstm_66_while_lstm_66_while_cond_437463915___redundant_placeholder2G
Clstm_66_while_lstm_66_while_cond_437463915___redundant_placeholder3
lstm_66_while_identity
�
lstm_66/while/LessLesslstm_66_while_placeholder*lstm_66_while_less_lstm_66_strided_slice_1*
T0*
_output_shapes
: [
lstm_66/while/IdentityIdentitylstm_66/while/Less:z:0*
T0
*
_output_shapes
: "9
lstm_66_while_identitylstm_66/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :���������d:���������d: ::::: 
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
:���������d:-)
'
_output_shapes
:���������d:

_output_shapes
: :

_output_shapes
:
�	
�
'__inference_signature_wrapper_437463810
lstm_66_input
unknown:	�
	unknown_0:	d�
	unknown_1:	�
	unknown_2:d2
	unknown_3:2
	unknown_4:22
	unknown_5:2
	unknown_6:2
	unknown_7:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalllstm_66_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*+
_read_only_resource_inputs
		*-
config_proto

CPU

GPU 2J 8� *-
f(R&
$__inference__wrapped_model_437462850o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):���������: : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
+
_output_shapes
:���������
'
_user_specified_namelstm_66_input
�K
�
F__inference_lstm_66_layer_call_and_return_conditional_losses_437463355

inputs>
+lstm_cell_66_matmul_readvariableop_resource:	�@
-lstm_cell_66_matmul_1_readvariableop_resource:	d�;
,lstm_cell_66_biasadd_readvariableop_resource:	�
identity��#lstm_cell_66/BiasAdd/ReadVariableOp�"lstm_cell_66/MatMul/ReadVariableOp�$lstm_cell_66/MatMul_1/ReadVariableOp�while;
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
valueB:�
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
value	B :ds
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
:���������dR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :dw
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
:���������dc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:���������D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
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
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
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
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
"lstm_cell_66/MatMul/ReadVariableOpReadVariableOp+lstm_cell_66_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
lstm_cell_66/MatMulMatMulstrided_slice_2:output:0*lstm_cell_66/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
$lstm_cell_66/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_66_matmul_1_readvariableop_resource*
_output_shapes
:	d�*
dtype0�
lstm_cell_66/MatMul_1MatMulzeros:output:0,lstm_cell_66/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
lstm_cell_66/addAddV2lstm_cell_66/MatMul:product:0lstm_cell_66/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
#lstm_cell_66/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_66_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
lstm_cell_66/BiasAddBiasAddlstm_cell_66/add:z:0+lstm_cell_66/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������^
lstm_cell_66/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_cell_66/splitSplit%lstm_cell_66/split/split_dim:output:0lstm_cell_66/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������d:���������d:���������d:���������d*
	num_splitn
lstm_cell_66/SigmoidSigmoidlstm_cell_66/split:output:0*
T0*'
_output_shapes
:���������dp
lstm_cell_66/Sigmoid_1Sigmoidlstm_cell_66/split:output:1*
T0*'
_output_shapes
:���������dw
lstm_cell_66/mulMullstm_cell_66/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������dh
lstm_cell_66/ReluRelulstm_cell_66/split:output:2*
T0*'
_output_shapes
:���������d�
lstm_cell_66/mul_1Mullstm_cell_66/Sigmoid:y:0lstm_cell_66/Relu:activations:0*
T0*'
_output_shapes
:���������d{
lstm_cell_66/add_1AddV2lstm_cell_66/mul:z:0lstm_cell_66/mul_1:z:0*
T0*'
_output_shapes
:���������dp
lstm_cell_66/Sigmoid_2Sigmoidlstm_cell_66/split:output:3*
T0*'
_output_shapes
:���������de
lstm_cell_66/Relu_1Relulstm_cell_66/add_1:z:0*
T0*'
_output_shapes
:���������d�
lstm_cell_66/mul_2Mullstm_cell_66/Sigmoid_2:y:0!lstm_cell_66/Relu_1:activations:0*
T0*'
_output_shapes
:���������dn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
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
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_66_matmul_readvariableop_resource-lstm_cell_66_matmul_1_readvariableop_resource,lstm_cell_66_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������d:���������d: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( * 
bodyR
while_body_437463270* 
condR
while_cond_437463269*K
output_shapes:
8: : : : :���������d:���������d: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������d*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������d*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:���������d[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:���������d�
NoOpNoOp$^lstm_cell_66/BiasAdd/ReadVariableOp#^lstm_cell_66/MatMul/ReadVariableOp%^lstm_cell_66/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������: : : 2J
#lstm_cell_66/BiasAdd/ReadVariableOp#lstm_cell_66/BiasAdd/ReadVariableOp2H
"lstm_cell_66/MatMul/ReadVariableOp"lstm_cell_66/MatMul/ReadVariableOp2L
$lstm_cell_66/MatMul_1/ReadVariableOp$lstm_cell_66/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�9
�
F__inference_lstm_66_layer_call_and_return_conditional_losses_437463002

inputs)
lstm_cell_66_437462918:	�)
lstm_cell_66_437462920:	d�%
lstm_cell_66_437462922:	�
identity��$lstm_cell_66/StatefulPartitionedCall�while;
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
valueB:�
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
value	B :ds
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
:���������dR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :dw
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
:���������dc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :������������������D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
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
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
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
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
$lstm_cell_66/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_66_437462918lstm_cell_66_437462920lstm_cell_66_437462922*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:���������d:���������d:���������d*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_lstm_cell_66_layer_call_and_return_conditional_losses_437462917n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
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
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_66_437462918lstm_cell_66_437462920lstm_cell_66_437462922*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������d:���������d: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( * 
bodyR
while_body_437462932* 
condR
while_cond_437462931*K
output_shapes:
8: : : : :���������d:���������d: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������d*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������d*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:���������d[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:���������du
NoOpNoOp%^lstm_cell_66/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 2L
$lstm_cell_66/StatefulPartitionedCall$lstm_cell_66/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs
�9
�
while_body_437464435
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_66_matmul_readvariableop_resource_0:	�H
5while_lstm_cell_66_matmul_1_readvariableop_resource_0:	d�C
4while_lstm_cell_66_biasadd_readvariableop_resource_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_66_matmul_readvariableop_resource:	�F
3while_lstm_cell_66_matmul_1_readvariableop_resource:	d�A
2while_lstm_cell_66_biasadd_readvariableop_resource:	���)while/lstm_cell_66/BiasAdd/ReadVariableOp�(while/lstm_cell_66/MatMul/ReadVariableOp�*while/lstm_cell_66/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
(while/lstm_cell_66/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_66_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
while/lstm_cell_66/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_66/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
*while/lstm_cell_66/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_66_matmul_1_readvariableop_resource_0*
_output_shapes
:	d�*
dtype0�
while/lstm_cell_66/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_66/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
while/lstm_cell_66/addAddV2#while/lstm_cell_66/MatMul:product:0%while/lstm_cell_66/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
)while/lstm_cell_66/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_66_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype0�
while/lstm_cell_66/BiasAddBiasAddwhile/lstm_cell_66/add:z:01while/lstm_cell_66/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������d
"while/lstm_cell_66/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/lstm_cell_66/splitSplit+while/lstm_cell_66/split/split_dim:output:0#while/lstm_cell_66/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������d:���������d:���������d:���������d*
	num_splitz
while/lstm_cell_66/SigmoidSigmoid!while/lstm_cell_66/split:output:0*
T0*'
_output_shapes
:���������d|
while/lstm_cell_66/Sigmoid_1Sigmoid!while/lstm_cell_66/split:output:1*
T0*'
_output_shapes
:���������d�
while/lstm_cell_66/mulMul while/lstm_cell_66/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������dt
while/lstm_cell_66/ReluRelu!while/lstm_cell_66/split:output:2*
T0*'
_output_shapes
:���������d�
while/lstm_cell_66/mul_1Mulwhile/lstm_cell_66/Sigmoid:y:0%while/lstm_cell_66/Relu:activations:0*
T0*'
_output_shapes
:���������d�
while/lstm_cell_66/add_1AddV2while/lstm_cell_66/mul:z:0while/lstm_cell_66/mul_1:z:0*
T0*'
_output_shapes
:���������d|
while/lstm_cell_66/Sigmoid_2Sigmoid!while/lstm_cell_66/split:output:3*
T0*'
_output_shapes
:���������dq
while/lstm_cell_66/Relu_1Reluwhile/lstm_cell_66/add_1:z:0*
T0*'
_output_shapes
:���������d�
while/lstm_cell_66/mul_2Mul while/lstm_cell_66/Sigmoid_2:y:0'while/lstm_cell_66/Relu_1:activations:0*
T0*'
_output_shapes
:���������dr
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : �
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell_66/mul_2:z:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: y
while/Identity_4Identitywhile/lstm_cell_66/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:���������dy
while/Identity_5Identitywhile/lstm_cell_66/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:���������d�

while/NoOpNoOp*^while/lstm_cell_66/BiasAdd/ReadVariableOp)^while/lstm_cell_66/MatMul/ReadVariableOp+^while/lstm_cell_66/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_66_biasadd_readvariableop_resource4while_lstm_cell_66_biasadd_readvariableop_resource_0"l
3while_lstm_cell_66_matmul_1_readvariableop_resource5while_lstm_cell_66_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_66_matmul_readvariableop_resource3while_lstm_cell_66_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������d:���������d: : : : : 2V
)while/lstm_cell_66/BiasAdd/ReadVariableOp)while/lstm_cell_66/BiasAdd/ReadVariableOp2T
(while/lstm_cell_66/MatMul/ReadVariableOp(while/lstm_cell_66/MatMul/ReadVariableOp2X
*while/lstm_cell_66/MatMul_1/ReadVariableOp*while/lstm_cell_66/MatMul_1/ReadVariableOp: 
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
:���������d:-)
'
_output_shapes
:���������d:

_output_shapes
: :

_output_shapes
: 
�9
�
while_body_437464580
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_66_matmul_readvariableop_resource_0:	�H
5while_lstm_cell_66_matmul_1_readvariableop_resource_0:	d�C
4while_lstm_cell_66_biasadd_readvariableop_resource_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_66_matmul_readvariableop_resource:	�F
3while_lstm_cell_66_matmul_1_readvariableop_resource:	d�A
2while_lstm_cell_66_biasadd_readvariableop_resource:	���)while/lstm_cell_66/BiasAdd/ReadVariableOp�(while/lstm_cell_66/MatMul/ReadVariableOp�*while/lstm_cell_66/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
(while/lstm_cell_66/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_66_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
while/lstm_cell_66/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_66/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
*while/lstm_cell_66/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_66_matmul_1_readvariableop_resource_0*
_output_shapes
:	d�*
dtype0�
while/lstm_cell_66/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_66/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
while/lstm_cell_66/addAddV2#while/lstm_cell_66/MatMul:product:0%while/lstm_cell_66/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
)while/lstm_cell_66/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_66_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype0�
while/lstm_cell_66/BiasAddBiasAddwhile/lstm_cell_66/add:z:01while/lstm_cell_66/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������d
"while/lstm_cell_66/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/lstm_cell_66/splitSplit+while/lstm_cell_66/split/split_dim:output:0#while/lstm_cell_66/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������d:���������d:���������d:���������d*
	num_splitz
while/lstm_cell_66/SigmoidSigmoid!while/lstm_cell_66/split:output:0*
T0*'
_output_shapes
:���������d|
while/lstm_cell_66/Sigmoid_1Sigmoid!while/lstm_cell_66/split:output:1*
T0*'
_output_shapes
:���������d�
while/lstm_cell_66/mulMul while/lstm_cell_66/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������dt
while/lstm_cell_66/ReluRelu!while/lstm_cell_66/split:output:2*
T0*'
_output_shapes
:���������d�
while/lstm_cell_66/mul_1Mulwhile/lstm_cell_66/Sigmoid:y:0%while/lstm_cell_66/Relu:activations:0*
T0*'
_output_shapes
:���������d�
while/lstm_cell_66/add_1AddV2while/lstm_cell_66/mul:z:0while/lstm_cell_66/mul_1:z:0*
T0*'
_output_shapes
:���������d|
while/lstm_cell_66/Sigmoid_2Sigmoid!while/lstm_cell_66/split:output:3*
T0*'
_output_shapes
:���������dq
while/lstm_cell_66/Relu_1Reluwhile/lstm_cell_66/add_1:z:0*
T0*'
_output_shapes
:���������d�
while/lstm_cell_66/mul_2Mul while/lstm_cell_66/Sigmoid_2:y:0'while/lstm_cell_66/Relu_1:activations:0*
T0*'
_output_shapes
:���������dr
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : �
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell_66/mul_2:z:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: y
while/Identity_4Identitywhile/lstm_cell_66/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:���������dy
while/Identity_5Identitywhile/lstm_cell_66/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:���������d�

while/NoOpNoOp*^while/lstm_cell_66/BiasAdd/ReadVariableOp)^while/lstm_cell_66/MatMul/ReadVariableOp+^while/lstm_cell_66/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_66_biasadd_readvariableop_resource4while_lstm_cell_66_biasadd_readvariableop_resource_0"l
3while_lstm_cell_66_matmul_1_readvariableop_resource5while_lstm_cell_66_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_66_matmul_readvariableop_resource3while_lstm_cell_66_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������d:���������d: : : : : 2V
)while/lstm_cell_66/BiasAdd/ReadVariableOp)while/lstm_cell_66/BiasAdd/ReadVariableOp2T
(while/lstm_cell_66/MatMul/ReadVariableOp(while/lstm_cell_66/MatMul/ReadVariableOp2X
*while/lstm_cell_66/MatMul_1/ReadVariableOp*while/lstm_cell_66/MatMul_1/ReadVariableOp: 
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
:���������d:-)
'
_output_shapes
:���������d:

_output_shapes
: :

_output_shapes
: 
�
�
K__inference_lstm_cell_66_layer_call_and_return_conditional_losses_437463065

inputs

states
states_11
matmul_readvariableop_resource:	�3
 matmul_1_readvariableop_resource:	d�.
biasadd_readvariableop_resource:	�
identity

identity_1

identity_2��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������y
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	d�*
dtype0n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������e
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:���������d:���������d:���������d:���������d*
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:���������dV
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:���������dU
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:���������dN
ReluRelusplit:output:2*
T0*'
_output_shapes
:���������d_
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:���������dT
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:���������dV
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:���������dK
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:���������dc
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:���������dX
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:���������dZ

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:���������dZ

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:���������d�
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������:���������d:���������d: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������d
 
_user_specified_namestates:OK
'
_output_shapes
:���������d
 
_user_specified_namestates
�
�
L__inference_sequential_66_layer_call_and_return_conditional_losses_437463414

inputs$
lstm_66_437463356:	�$
lstm_66_437463358:	d� 
lstm_66_437463360:	�%
dense_198_437463375:d2!
dense_198_437463377:2%
dense_199_437463392:22!
dense_199_437463394:2%
dense_200_437463408:2!
dense_200_437463410:
identity��!dense_198/StatefulPartitionedCall�!dense_199/StatefulPartitionedCall�!dense_200/StatefulPartitionedCall�lstm_66/StatefulPartitionedCall�
lstm_66/StatefulPartitionedCallStatefulPartitionedCallinputslstm_66_437463356lstm_66_437463358lstm_66_437463360*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_lstm_66_layer_call_and_return_conditional_losses_437463355�
!dense_198/StatefulPartitionedCallStatefulPartitionedCall(lstm_66/StatefulPartitionedCall:output:0dense_198_437463375dense_198_437463377*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_dense_198_layer_call_and_return_conditional_losses_437463374�
!dense_199/StatefulPartitionedCallStatefulPartitionedCall*dense_198/StatefulPartitionedCall:output:0dense_199_437463392dense_199_437463394*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_dense_199_layer_call_and_return_conditional_losses_437463391�
!dense_200/StatefulPartitionedCallStatefulPartitionedCall*dense_199/StatefulPartitionedCall:output:0dense_200_437463408dense_200_437463410*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_dense_200_layer_call_and_return_conditional_losses_437463407y
IdentityIdentity*dense_200/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp"^dense_198/StatefulPartitionedCall"^dense_199/StatefulPartitionedCall"^dense_200/StatefulPartitionedCall ^lstm_66/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):���������: : : : : : : : : 2F
!dense_198/StatefulPartitionedCall!dense_198/StatefulPartitionedCall2F
!dense_199/StatefulPartitionedCall!dense_199/StatefulPartitionedCall2F
!dense_200/StatefulPartitionedCall!dense_200/StatefulPartitionedCall2B
lstm_66/StatefulPartitionedCalllstm_66/StatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
K__inference_lstm_cell_66_layer_call_and_return_conditional_losses_437464967

inputs
states_0
states_11
matmul_readvariableop_resource:	�3
 matmul_1_readvariableop_resource:	d�.
biasadd_readvariableop_resource:	�
identity

identity_1

identity_2��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������y
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	d�*
dtype0p
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������e
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:���������d:���������d:���������d:���������d*
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:���������dV
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:���������dU
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:���������dN
ReluRelusplit:output:2*
T0*'
_output_shapes
:���������d_
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:���������dT
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:���������dV
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:���������dK
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:���������dc
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:���������dX
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:���������dZ

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:���������dZ

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:���������d�
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������:���������d:���������d: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:QM
'
_output_shapes
:���������d
"
_user_specified_name
states/0:QM
'
_output_shapes
:���������d
"
_user_specified_name
states/1
�
�
while_cond_437464434
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_17
3while_while_cond_437464434___redundant_placeholder07
3while_while_cond_437464434___redundant_placeholder17
3while_while_cond_437464434___redundant_placeholder27
3while_while_cond_437464434___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
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
@: : : : :���������d:���������d: ::::: 
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
:���������d:-)
'
_output_shapes
:���������d:

_output_shapes
: :

_output_shapes
:
�
�
L__inference_sequential_66_layer_call_and_return_conditional_losses_437463753
lstm_66_input$
lstm_66_437463730:	�$
lstm_66_437463732:	d� 
lstm_66_437463734:	�%
dense_198_437463737:d2!
dense_198_437463739:2%
dense_199_437463742:22!
dense_199_437463744:2%
dense_200_437463747:2!
dense_200_437463749:
identity��!dense_198/StatefulPartitionedCall�!dense_199/StatefulPartitionedCall�!dense_200/StatefulPartitionedCall�lstm_66/StatefulPartitionedCall�
lstm_66/StatefulPartitionedCallStatefulPartitionedCalllstm_66_inputlstm_66_437463730lstm_66_437463732lstm_66_437463734*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_lstm_66_layer_call_and_return_conditional_losses_437463355�
!dense_198/StatefulPartitionedCallStatefulPartitionedCall(lstm_66/StatefulPartitionedCall:output:0dense_198_437463737dense_198_437463739*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_dense_198_layer_call_and_return_conditional_losses_437463374�
!dense_199/StatefulPartitionedCallStatefulPartitionedCall*dense_198/StatefulPartitionedCall:output:0dense_199_437463742dense_199_437463744*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_dense_199_layer_call_and_return_conditional_losses_437463391�
!dense_200/StatefulPartitionedCallStatefulPartitionedCall*dense_199/StatefulPartitionedCall:output:0dense_200_437463747dense_200_437463749*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_dense_200_layer_call_and_return_conditional_losses_437463407y
IdentityIdentity*dense_200/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp"^dense_198/StatefulPartitionedCall"^dense_199/StatefulPartitionedCall"^dense_200/StatefulPartitionedCall ^lstm_66/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):���������: : : : : : : : : 2F
!dense_198/StatefulPartitionedCall!dense_198/StatefulPartitionedCall2F
!dense_199/StatefulPartitionedCall!dense_199/StatefulPartitionedCall2F
!dense_200/StatefulPartitionedCall!dense_200/StatefulPartitionedCall2B
lstm_66/StatefulPartitionedCalllstm_66/StatefulPartitionedCall:Z V
+
_output_shapes
:���������
'
_user_specified_namelstm_66_input
�

�
1__inference_sequential_66_layer_call_fn_437463727
lstm_66_input
unknown:	�
	unknown_0:	d�
	unknown_1:	�
	unknown_2:d2
	unknown_3:2
	unknown_4:22
	unknown_5:2
	unknown_6:2
	unknown_7:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalllstm_66_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*+
_read_only_resource_inputs
		*-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_sequential_66_layer_call_and_return_conditional_losses_437463683o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):���������: : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
+
_output_shapes
:���������
'
_user_specified_namelstm_66_input
�
�
+__inference_lstm_66_layer_call_fn_437464197
inputs_0
unknown:	�
	unknown_0:	d�
	unknown_1:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_lstm_66_layer_call_and_return_conditional_losses_437463002o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :������������������
"
_user_specified_name
inputs/0
�

�
H__inference_dense_198_layer_call_and_return_conditional_losses_437463374

inputs0
matmul_readvariableop_resource:d2-
biasadd_readvariableop_resource:2
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d2*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������2a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������2w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
��
�
%__inference__traced_restore_437465216
file_prefix3
!assignvariableop_dense_198_kernel:d2/
!assignvariableop_1_dense_198_bias:25
#assignvariableop_2_dense_199_kernel:22/
!assignvariableop_3_dense_199_bias:25
#assignvariableop_4_dense_200_kernel:2/
!assignvariableop_5_dense_200_bias:A
.assignvariableop_6_lstm_66_lstm_cell_66_kernel:	�K
8assignvariableop_7_lstm_66_lstm_cell_66_recurrent_kernel:	d�;
,assignvariableop_8_lstm_66_lstm_cell_66_bias:	�&
assignvariableop_9_adam_iter:	 )
assignvariableop_10_adam_beta_1: )
assignvariableop_11_adam_beta_2: (
assignvariableop_12_adam_decay: 0
&assignvariableop_13_adam_learning_rate: %
assignvariableop_14_total_1: %
assignvariableop_15_count_1: #
assignvariableop_16_total: #
assignvariableop_17_count: =
+assignvariableop_18_adam_dense_198_kernel_m:d27
)assignvariableop_19_adam_dense_198_bias_m:2=
+assignvariableop_20_adam_dense_199_kernel_m:227
)assignvariableop_21_adam_dense_199_bias_m:2=
+assignvariableop_22_adam_dense_200_kernel_m:27
)assignvariableop_23_adam_dense_200_bias_m:I
6assignvariableop_24_adam_lstm_66_lstm_cell_66_kernel_m:	�S
@assignvariableop_25_adam_lstm_66_lstm_cell_66_recurrent_kernel_m:	d�C
4assignvariableop_26_adam_lstm_66_lstm_cell_66_bias_m:	�=
+assignvariableop_27_adam_dense_198_kernel_v:d27
)assignvariableop_28_adam_dense_198_bias_v:2=
+assignvariableop_29_adam_dense_199_kernel_v:227
)assignvariableop_30_adam_dense_199_bias_v:2=
+assignvariableop_31_adam_dense_200_kernel_v:27
)assignvariableop_32_adam_dense_200_bias_v:I
6assignvariableop_33_adam_lstm_66_lstm_cell_66_kernel_v:	�S
@assignvariableop_34_adam_lstm_66_lstm_cell_66_recurrent_kernel_v:	d�C
4assignvariableop_35_adam_lstm_66_lstm_cell_66_bias_v:	�
identity_37��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:%*
dtype0*�
value�B�%B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:%*
dtype0*]
valueTBR%B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�:::::::::::::::::::::::::::::::::::::*3
dtypes)
'2%	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOp!assignvariableop_dense_198_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp!assignvariableop_1_dense_198_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp#assignvariableop_2_dense_199_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp!assignvariableop_3_dense_199_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp#assignvariableop_4_dense_200_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp!assignvariableop_5_dense_200_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp.assignvariableop_6_lstm_66_lstm_cell_66_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp8assignvariableop_7_lstm_66_lstm_cell_66_recurrent_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp,assignvariableop_8_lstm_66_lstm_cell_66_biasIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_iterIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_beta_1Identity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_beta_2Identity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOpassignvariableop_12_adam_decayIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp&assignvariableop_13_adam_learning_rateIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOpassignvariableop_14_total_1Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOpassignvariableop_15_count_1Identity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOpassignvariableop_16_totalIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOpassignvariableop_17_countIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp+assignvariableop_18_adam_dense_198_kernel_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp)assignvariableop_19_adam_dense_198_bias_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOp+assignvariableop_20_adam_dense_199_kernel_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOp)assignvariableop_21_adam_dense_199_bias_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOp+assignvariableop_22_adam_dense_200_kernel_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOp)assignvariableop_23_adam_dense_200_bias_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOp6assignvariableop_24_adam_lstm_66_lstm_cell_66_kernel_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOp@assignvariableop_25_adam_lstm_66_lstm_cell_66_recurrent_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOp4assignvariableop_26_adam_lstm_66_lstm_cell_66_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOp+assignvariableop_27_adam_dense_198_kernel_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOp)assignvariableop_28_adam_dense_198_bias_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOp+assignvariableop_29_adam_dense_199_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_30AssignVariableOp)assignvariableop_30_adam_dense_199_bias_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_31AssignVariableOp+assignvariableop_31_adam_dense_200_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_32AssignVariableOp)assignvariableop_32_adam_dense_200_bias_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_33AssignVariableOp6assignvariableop_33_adam_lstm_66_lstm_cell_66_kernel_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_34AssignVariableOp@assignvariableop_34_adam_lstm_66_lstm_cell_66_recurrent_kernel_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_35AssignVariableOp4assignvariableop_35_adam_lstm_66_lstm_cell_66_bias_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 �
Identity_36Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_37IdentityIdentity_36:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_37Identity_37:output:0*]
_input_shapesL
J: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352(
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
�
�
while_cond_437464579
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_17
3while_while_cond_437464579___redundant_placeholder07
3while_while_cond_437464579___redundant_placeholder17
3while_while_cond_437464579___redundant_placeholder27
3while_while_cond_437464579___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
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
@: : : : :���������d:���������d: ::::: 
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
:���������d:-)
'
_output_shapes
:���������d:

_output_shapes
: :

_output_shapes
:
�9
�
while_body_437463270
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_66_matmul_readvariableop_resource_0:	�H
5while_lstm_cell_66_matmul_1_readvariableop_resource_0:	d�C
4while_lstm_cell_66_biasadd_readvariableop_resource_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_66_matmul_readvariableop_resource:	�F
3while_lstm_cell_66_matmul_1_readvariableop_resource:	d�A
2while_lstm_cell_66_biasadd_readvariableop_resource:	���)while/lstm_cell_66/BiasAdd/ReadVariableOp�(while/lstm_cell_66/MatMul/ReadVariableOp�*while/lstm_cell_66/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
(while/lstm_cell_66/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_66_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
while/lstm_cell_66/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_66/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
*while/lstm_cell_66/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_66_matmul_1_readvariableop_resource_0*
_output_shapes
:	d�*
dtype0�
while/lstm_cell_66/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_66/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
while/lstm_cell_66/addAddV2#while/lstm_cell_66/MatMul:product:0%while/lstm_cell_66/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
)while/lstm_cell_66/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_66_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype0�
while/lstm_cell_66/BiasAddBiasAddwhile/lstm_cell_66/add:z:01while/lstm_cell_66/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������d
"while/lstm_cell_66/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/lstm_cell_66/splitSplit+while/lstm_cell_66/split/split_dim:output:0#while/lstm_cell_66/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������d:���������d:���������d:���������d*
	num_splitz
while/lstm_cell_66/SigmoidSigmoid!while/lstm_cell_66/split:output:0*
T0*'
_output_shapes
:���������d|
while/lstm_cell_66/Sigmoid_1Sigmoid!while/lstm_cell_66/split:output:1*
T0*'
_output_shapes
:���������d�
while/lstm_cell_66/mulMul while/lstm_cell_66/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������dt
while/lstm_cell_66/ReluRelu!while/lstm_cell_66/split:output:2*
T0*'
_output_shapes
:���������d�
while/lstm_cell_66/mul_1Mulwhile/lstm_cell_66/Sigmoid:y:0%while/lstm_cell_66/Relu:activations:0*
T0*'
_output_shapes
:���������d�
while/lstm_cell_66/add_1AddV2while/lstm_cell_66/mul:z:0while/lstm_cell_66/mul_1:z:0*
T0*'
_output_shapes
:���������d|
while/lstm_cell_66/Sigmoid_2Sigmoid!while/lstm_cell_66/split:output:3*
T0*'
_output_shapes
:���������dq
while/lstm_cell_66/Relu_1Reluwhile/lstm_cell_66/add_1:z:0*
T0*'
_output_shapes
:���������d�
while/lstm_cell_66/mul_2Mul while/lstm_cell_66/Sigmoid_2:y:0'while/lstm_cell_66/Relu_1:activations:0*
T0*'
_output_shapes
:���������dr
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : �
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell_66/mul_2:z:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: y
while/Identity_4Identitywhile/lstm_cell_66/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:���������dy
while/Identity_5Identitywhile/lstm_cell_66/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:���������d�

while/NoOpNoOp*^while/lstm_cell_66/BiasAdd/ReadVariableOp)^while/lstm_cell_66/MatMul/ReadVariableOp+^while/lstm_cell_66/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_66_biasadd_readvariableop_resource4while_lstm_cell_66_biasadd_readvariableop_resource_0"l
3while_lstm_cell_66_matmul_1_readvariableop_resource5while_lstm_cell_66_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_66_matmul_readvariableop_resource3while_lstm_cell_66_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������d:���������d: : : : : 2V
)while/lstm_cell_66/BiasAdd/ReadVariableOp)while/lstm_cell_66/BiasAdd/ReadVariableOp2T
(while/lstm_cell_66/MatMul/ReadVariableOp(while/lstm_cell_66/MatMul/ReadVariableOp2X
*while/lstm_cell_66/MatMul_1/ReadVariableOp*while/lstm_cell_66/MatMul_1/ReadVariableOp: 
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
:���������d:-)
'
_output_shapes
:���������d:

_output_shapes
: :

_output_shapes
: 
�
�
*sequential_66_lstm_66_while_cond_437462744H
Dsequential_66_lstm_66_while_sequential_66_lstm_66_while_loop_counterN
Jsequential_66_lstm_66_while_sequential_66_lstm_66_while_maximum_iterations+
'sequential_66_lstm_66_while_placeholder-
)sequential_66_lstm_66_while_placeholder_1-
)sequential_66_lstm_66_while_placeholder_2-
)sequential_66_lstm_66_while_placeholder_3J
Fsequential_66_lstm_66_while_less_sequential_66_lstm_66_strided_slice_1c
_sequential_66_lstm_66_while_sequential_66_lstm_66_while_cond_437462744___redundant_placeholder0c
_sequential_66_lstm_66_while_sequential_66_lstm_66_while_cond_437462744___redundant_placeholder1c
_sequential_66_lstm_66_while_sequential_66_lstm_66_while_cond_437462744___redundant_placeholder2c
_sequential_66_lstm_66_while_sequential_66_lstm_66_while_cond_437462744___redundant_placeholder3(
$sequential_66_lstm_66_while_identity
�
 sequential_66/lstm_66/while/LessLess'sequential_66_lstm_66_while_placeholderFsequential_66_lstm_66_while_less_sequential_66_lstm_66_strided_slice_1*
T0*
_output_shapes
: w
$sequential_66/lstm_66/while/IdentityIdentity$sequential_66/lstm_66/while/Less:z:0*
T0
*
_output_shapes
: "U
$sequential_66_lstm_66_while_identity-sequential_66/lstm_66/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :���������d:���������d: ::::: 
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
:���������d:-)
'
_output_shapes
:���������d:

_output_shapes
: :

_output_shapes
:
�S
�
*sequential_66_lstm_66_while_body_437462745H
Dsequential_66_lstm_66_while_sequential_66_lstm_66_while_loop_counterN
Jsequential_66_lstm_66_while_sequential_66_lstm_66_while_maximum_iterations+
'sequential_66_lstm_66_while_placeholder-
)sequential_66_lstm_66_while_placeholder_1-
)sequential_66_lstm_66_while_placeholder_2-
)sequential_66_lstm_66_while_placeholder_3G
Csequential_66_lstm_66_while_sequential_66_lstm_66_strided_slice_1_0�
sequential_66_lstm_66_while_tensorarrayv2read_tensorlistgetitem_sequential_66_lstm_66_tensorarrayunstack_tensorlistfromtensor_0\
Isequential_66_lstm_66_while_lstm_cell_66_matmul_readvariableop_resource_0:	�^
Ksequential_66_lstm_66_while_lstm_cell_66_matmul_1_readvariableop_resource_0:	d�Y
Jsequential_66_lstm_66_while_lstm_cell_66_biasadd_readvariableop_resource_0:	�(
$sequential_66_lstm_66_while_identity*
&sequential_66_lstm_66_while_identity_1*
&sequential_66_lstm_66_while_identity_2*
&sequential_66_lstm_66_while_identity_3*
&sequential_66_lstm_66_while_identity_4*
&sequential_66_lstm_66_while_identity_5E
Asequential_66_lstm_66_while_sequential_66_lstm_66_strided_slice_1�
}sequential_66_lstm_66_while_tensorarrayv2read_tensorlistgetitem_sequential_66_lstm_66_tensorarrayunstack_tensorlistfromtensorZ
Gsequential_66_lstm_66_while_lstm_cell_66_matmul_readvariableop_resource:	�\
Isequential_66_lstm_66_while_lstm_cell_66_matmul_1_readvariableop_resource:	d�W
Hsequential_66_lstm_66_while_lstm_cell_66_biasadd_readvariableop_resource:	���?sequential_66/lstm_66/while/lstm_cell_66/BiasAdd/ReadVariableOp�>sequential_66/lstm_66/while/lstm_cell_66/MatMul/ReadVariableOp�@sequential_66/lstm_66/while/lstm_cell_66/MatMul_1/ReadVariableOp�
Msequential_66/lstm_66/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
?sequential_66/lstm_66/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsequential_66_lstm_66_while_tensorarrayv2read_tensorlistgetitem_sequential_66_lstm_66_tensorarrayunstack_tensorlistfromtensor_0'sequential_66_lstm_66_while_placeholderVsequential_66/lstm_66/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
>sequential_66/lstm_66/while/lstm_cell_66/MatMul/ReadVariableOpReadVariableOpIsequential_66_lstm_66_while_lstm_cell_66_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
/sequential_66/lstm_66/while/lstm_cell_66/MatMulMatMulFsequential_66/lstm_66/while/TensorArrayV2Read/TensorListGetItem:item:0Fsequential_66/lstm_66/while/lstm_cell_66/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
@sequential_66/lstm_66/while/lstm_cell_66/MatMul_1/ReadVariableOpReadVariableOpKsequential_66_lstm_66_while_lstm_cell_66_matmul_1_readvariableop_resource_0*
_output_shapes
:	d�*
dtype0�
1sequential_66/lstm_66/while/lstm_cell_66/MatMul_1MatMul)sequential_66_lstm_66_while_placeholder_2Hsequential_66/lstm_66/while/lstm_cell_66/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
,sequential_66/lstm_66/while/lstm_cell_66/addAddV29sequential_66/lstm_66/while/lstm_cell_66/MatMul:product:0;sequential_66/lstm_66/while/lstm_cell_66/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
?sequential_66/lstm_66/while/lstm_cell_66/BiasAdd/ReadVariableOpReadVariableOpJsequential_66_lstm_66_while_lstm_cell_66_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype0�
0sequential_66/lstm_66/while/lstm_cell_66/BiasAddBiasAdd0sequential_66/lstm_66/while/lstm_cell_66/add:z:0Gsequential_66/lstm_66/while/lstm_cell_66/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������z
8sequential_66/lstm_66/while/lstm_cell_66/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
.sequential_66/lstm_66/while/lstm_cell_66/splitSplitAsequential_66/lstm_66/while/lstm_cell_66/split/split_dim:output:09sequential_66/lstm_66/while/lstm_cell_66/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������d:���������d:���������d:���������d*
	num_split�
0sequential_66/lstm_66/while/lstm_cell_66/SigmoidSigmoid7sequential_66/lstm_66/while/lstm_cell_66/split:output:0*
T0*'
_output_shapes
:���������d�
2sequential_66/lstm_66/while/lstm_cell_66/Sigmoid_1Sigmoid7sequential_66/lstm_66/while/lstm_cell_66/split:output:1*
T0*'
_output_shapes
:���������d�
,sequential_66/lstm_66/while/lstm_cell_66/mulMul6sequential_66/lstm_66/while/lstm_cell_66/Sigmoid_1:y:0)sequential_66_lstm_66_while_placeholder_3*
T0*'
_output_shapes
:���������d�
-sequential_66/lstm_66/while/lstm_cell_66/ReluRelu7sequential_66/lstm_66/while/lstm_cell_66/split:output:2*
T0*'
_output_shapes
:���������d�
.sequential_66/lstm_66/while/lstm_cell_66/mul_1Mul4sequential_66/lstm_66/while/lstm_cell_66/Sigmoid:y:0;sequential_66/lstm_66/while/lstm_cell_66/Relu:activations:0*
T0*'
_output_shapes
:���������d�
.sequential_66/lstm_66/while/lstm_cell_66/add_1AddV20sequential_66/lstm_66/while/lstm_cell_66/mul:z:02sequential_66/lstm_66/while/lstm_cell_66/mul_1:z:0*
T0*'
_output_shapes
:���������d�
2sequential_66/lstm_66/while/lstm_cell_66/Sigmoid_2Sigmoid7sequential_66/lstm_66/while/lstm_cell_66/split:output:3*
T0*'
_output_shapes
:���������d�
/sequential_66/lstm_66/while/lstm_cell_66/Relu_1Relu2sequential_66/lstm_66/while/lstm_cell_66/add_1:z:0*
T0*'
_output_shapes
:���������d�
.sequential_66/lstm_66/while/lstm_cell_66/mul_2Mul6sequential_66/lstm_66/while/lstm_cell_66/Sigmoid_2:y:0=sequential_66/lstm_66/while/lstm_cell_66/Relu_1:activations:0*
T0*'
_output_shapes
:���������d�
Fsequential_66/lstm_66/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : �
@sequential_66/lstm_66/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem)sequential_66_lstm_66_while_placeholder_1Osequential_66/lstm_66/while/TensorArrayV2Write/TensorListSetItem/index:output:02sequential_66/lstm_66/while/lstm_cell_66/mul_2:z:0*
_output_shapes
: *
element_dtype0:���c
!sequential_66/lstm_66/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :�
sequential_66/lstm_66/while/addAddV2'sequential_66_lstm_66_while_placeholder*sequential_66/lstm_66/while/add/y:output:0*
T0*
_output_shapes
: e
#sequential_66/lstm_66/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
!sequential_66/lstm_66/while/add_1AddV2Dsequential_66_lstm_66_while_sequential_66_lstm_66_while_loop_counter,sequential_66/lstm_66/while/add_1/y:output:0*
T0*
_output_shapes
: �
$sequential_66/lstm_66/while/IdentityIdentity%sequential_66/lstm_66/while/add_1:z:0!^sequential_66/lstm_66/while/NoOp*
T0*
_output_shapes
: �
&sequential_66/lstm_66/while/Identity_1IdentityJsequential_66_lstm_66_while_sequential_66_lstm_66_while_maximum_iterations!^sequential_66/lstm_66/while/NoOp*
T0*
_output_shapes
: �
&sequential_66/lstm_66/while/Identity_2Identity#sequential_66/lstm_66/while/add:z:0!^sequential_66/lstm_66/while/NoOp*
T0*
_output_shapes
: �
&sequential_66/lstm_66/while/Identity_3IdentityPsequential_66/lstm_66/while/TensorArrayV2Write/TensorListSetItem:output_handle:0!^sequential_66/lstm_66/while/NoOp*
T0*
_output_shapes
: �
&sequential_66/lstm_66/while/Identity_4Identity2sequential_66/lstm_66/while/lstm_cell_66/mul_2:z:0!^sequential_66/lstm_66/while/NoOp*
T0*'
_output_shapes
:���������d�
&sequential_66/lstm_66/while/Identity_5Identity2sequential_66/lstm_66/while/lstm_cell_66/add_1:z:0!^sequential_66/lstm_66/while/NoOp*
T0*'
_output_shapes
:���������d�
 sequential_66/lstm_66/while/NoOpNoOp@^sequential_66/lstm_66/while/lstm_cell_66/BiasAdd/ReadVariableOp?^sequential_66/lstm_66/while/lstm_cell_66/MatMul/ReadVariableOpA^sequential_66/lstm_66/while/lstm_cell_66/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "U
$sequential_66_lstm_66_while_identity-sequential_66/lstm_66/while/Identity:output:0"Y
&sequential_66_lstm_66_while_identity_1/sequential_66/lstm_66/while/Identity_1:output:0"Y
&sequential_66_lstm_66_while_identity_2/sequential_66/lstm_66/while/Identity_2:output:0"Y
&sequential_66_lstm_66_while_identity_3/sequential_66/lstm_66/while/Identity_3:output:0"Y
&sequential_66_lstm_66_while_identity_4/sequential_66/lstm_66/while/Identity_4:output:0"Y
&sequential_66_lstm_66_while_identity_5/sequential_66/lstm_66/while/Identity_5:output:0"�
Hsequential_66_lstm_66_while_lstm_cell_66_biasadd_readvariableop_resourceJsequential_66_lstm_66_while_lstm_cell_66_biasadd_readvariableop_resource_0"�
Isequential_66_lstm_66_while_lstm_cell_66_matmul_1_readvariableop_resourceKsequential_66_lstm_66_while_lstm_cell_66_matmul_1_readvariableop_resource_0"�
Gsequential_66_lstm_66_while_lstm_cell_66_matmul_readvariableop_resourceIsequential_66_lstm_66_while_lstm_cell_66_matmul_readvariableop_resource_0"�
Asequential_66_lstm_66_while_sequential_66_lstm_66_strided_slice_1Csequential_66_lstm_66_while_sequential_66_lstm_66_strided_slice_1_0"�
}sequential_66_lstm_66_while_tensorarrayv2read_tensorlistgetitem_sequential_66_lstm_66_tensorarrayunstack_tensorlistfromtensorsequential_66_lstm_66_while_tensorarrayv2read_tensorlistgetitem_sequential_66_lstm_66_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������d:���������d: : : : : 2�
?sequential_66/lstm_66/while/lstm_cell_66/BiasAdd/ReadVariableOp?sequential_66/lstm_66/while/lstm_cell_66/BiasAdd/ReadVariableOp2�
>sequential_66/lstm_66/while/lstm_cell_66/MatMul/ReadVariableOp>sequential_66/lstm_66/while/lstm_cell_66/MatMul/ReadVariableOp2�
@sequential_66/lstm_66/while/lstm_cell_66/MatMul_1/ReadVariableOp@sequential_66/lstm_66/while/lstm_cell_66/MatMul_1/ReadVariableOp: 
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
:���������d:-)
'
_output_shapes
:���������d:

_output_shapes
: :

_output_shapes
: 
�m
�
L__inference_sequential_66_layer_call_and_return_conditional_losses_437464186

inputsF
3lstm_66_lstm_cell_66_matmul_readvariableop_resource:	�H
5lstm_66_lstm_cell_66_matmul_1_readvariableop_resource:	d�C
4lstm_66_lstm_cell_66_biasadd_readvariableop_resource:	�:
(dense_198_matmul_readvariableop_resource:d27
)dense_198_biasadd_readvariableop_resource:2:
(dense_199_matmul_readvariableop_resource:227
)dense_199_biasadd_readvariableop_resource:2:
(dense_200_matmul_readvariableop_resource:27
)dense_200_biasadd_readvariableop_resource:
identity�� dense_198/BiasAdd/ReadVariableOp�dense_198/MatMul/ReadVariableOp� dense_199/BiasAdd/ReadVariableOp�dense_199/MatMul/ReadVariableOp� dense_200/BiasAdd/ReadVariableOp�dense_200/MatMul/ReadVariableOp�+lstm_66/lstm_cell_66/BiasAdd/ReadVariableOp�*lstm_66/lstm_cell_66/MatMul/ReadVariableOp�,lstm_66/lstm_cell_66/MatMul_1/ReadVariableOp�lstm_66/whileC
lstm_66/ShapeShapeinputs*
T0*
_output_shapes
:e
lstm_66/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
lstm_66/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
lstm_66/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
lstm_66/strided_sliceStridedSlicelstm_66/Shape:output:0$lstm_66/strided_slice/stack:output:0&lstm_66/strided_slice/stack_1:output:0&lstm_66/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
lstm_66/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d�
lstm_66/zeros/packedPacklstm_66/strided_slice:output:0lstm_66/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:X
lstm_66/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
lstm_66/zerosFilllstm_66/zeros/packed:output:0lstm_66/zeros/Const:output:0*
T0*'
_output_shapes
:���������dZ
lstm_66/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d�
lstm_66/zeros_1/packedPacklstm_66/strided_slice:output:0!lstm_66/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:Z
lstm_66/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
lstm_66/zeros_1Filllstm_66/zeros_1/packed:output:0lstm_66/zeros_1/Const:output:0*
T0*'
_output_shapes
:���������dk
lstm_66/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          }
lstm_66/transpose	Transposeinputslstm_66/transpose/perm:output:0*
T0*+
_output_shapes
:���������T
lstm_66/Shape_1Shapelstm_66/transpose:y:0*
T0*
_output_shapes
:g
lstm_66/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: i
lstm_66/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
lstm_66/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
lstm_66/strided_slice_1StridedSlicelstm_66/Shape_1:output:0&lstm_66/strided_slice_1/stack:output:0(lstm_66/strided_slice_1/stack_1:output:0(lstm_66/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskn
#lstm_66/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
lstm_66/TensorArrayV2TensorListReserve,lstm_66/TensorArrayV2/element_shape:output:0 lstm_66/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
=lstm_66/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
/lstm_66/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_66/transpose:y:0Flstm_66/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���g
lstm_66/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: i
lstm_66/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
lstm_66/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
lstm_66/strided_slice_2StridedSlicelstm_66/transpose:y:0&lstm_66/strided_slice_2/stack:output:0(lstm_66/strided_slice_2/stack_1:output:0(lstm_66/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
*lstm_66/lstm_cell_66/MatMul/ReadVariableOpReadVariableOp3lstm_66_lstm_cell_66_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
lstm_66/lstm_cell_66/MatMulMatMul lstm_66/strided_slice_2:output:02lstm_66/lstm_cell_66/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
,lstm_66/lstm_cell_66/MatMul_1/ReadVariableOpReadVariableOp5lstm_66_lstm_cell_66_matmul_1_readvariableop_resource*
_output_shapes
:	d�*
dtype0�
lstm_66/lstm_cell_66/MatMul_1MatMullstm_66/zeros:output:04lstm_66/lstm_cell_66/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
lstm_66/lstm_cell_66/addAddV2%lstm_66/lstm_cell_66/MatMul:product:0'lstm_66/lstm_cell_66/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
+lstm_66/lstm_cell_66/BiasAdd/ReadVariableOpReadVariableOp4lstm_66_lstm_cell_66_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
lstm_66/lstm_cell_66/BiasAddBiasAddlstm_66/lstm_cell_66/add:z:03lstm_66/lstm_cell_66/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������f
$lstm_66/lstm_cell_66/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_66/lstm_cell_66/splitSplit-lstm_66/lstm_cell_66/split/split_dim:output:0%lstm_66/lstm_cell_66/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������d:���������d:���������d:���������d*
	num_split~
lstm_66/lstm_cell_66/SigmoidSigmoid#lstm_66/lstm_cell_66/split:output:0*
T0*'
_output_shapes
:���������d�
lstm_66/lstm_cell_66/Sigmoid_1Sigmoid#lstm_66/lstm_cell_66/split:output:1*
T0*'
_output_shapes
:���������d�
lstm_66/lstm_cell_66/mulMul"lstm_66/lstm_cell_66/Sigmoid_1:y:0lstm_66/zeros_1:output:0*
T0*'
_output_shapes
:���������dx
lstm_66/lstm_cell_66/ReluRelu#lstm_66/lstm_cell_66/split:output:2*
T0*'
_output_shapes
:���������d�
lstm_66/lstm_cell_66/mul_1Mul lstm_66/lstm_cell_66/Sigmoid:y:0'lstm_66/lstm_cell_66/Relu:activations:0*
T0*'
_output_shapes
:���������d�
lstm_66/lstm_cell_66/add_1AddV2lstm_66/lstm_cell_66/mul:z:0lstm_66/lstm_cell_66/mul_1:z:0*
T0*'
_output_shapes
:���������d�
lstm_66/lstm_cell_66/Sigmoid_2Sigmoid#lstm_66/lstm_cell_66/split:output:3*
T0*'
_output_shapes
:���������du
lstm_66/lstm_cell_66/Relu_1Relulstm_66/lstm_cell_66/add_1:z:0*
T0*'
_output_shapes
:���������d�
lstm_66/lstm_cell_66/mul_2Mul"lstm_66/lstm_cell_66/Sigmoid_2:y:0)lstm_66/lstm_cell_66/Relu_1:activations:0*
T0*'
_output_shapes
:���������dv
%lstm_66/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   f
$lstm_66/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_66/TensorArrayV2_1TensorListReserve.lstm_66/TensorArrayV2_1/element_shape:output:0-lstm_66/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���N
lstm_66/timeConst*
_output_shapes
: *
dtype0*
value	B : k
 lstm_66/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������\
lstm_66/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
lstm_66/whileWhile#lstm_66/while/loop_counter:output:0)lstm_66/while/maximum_iterations:output:0lstm_66/time:output:0 lstm_66/TensorArrayV2_1:handle:0lstm_66/zeros:output:0lstm_66/zeros_1:output:0 lstm_66/strided_slice_1:output:0?lstm_66/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_66_lstm_cell_66_matmul_readvariableop_resource5lstm_66_lstm_cell_66_matmul_1_readvariableop_resource4lstm_66_lstm_cell_66_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������d:���������d: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *(
body R
lstm_66_while_body_437464081*(
cond R
lstm_66_while_cond_437464080*K
output_shapes:
8: : : : :���������d:���������d: : : : : *
parallel_iterations �
8lstm_66/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   �
*lstm_66/TensorArrayV2Stack/TensorListStackTensorListStacklstm_66/while:output:3Alstm_66/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������d*
element_dtype0*
num_elementsp
lstm_66/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������i
lstm_66/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: i
lstm_66/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
lstm_66/strided_slice_3StridedSlice3lstm_66/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_66/strided_slice_3/stack:output:0(lstm_66/strided_slice_3/stack_1:output:0(lstm_66/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������d*
shrink_axis_maskm
lstm_66/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
lstm_66/transpose_1	Transpose3lstm_66/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_66/transpose_1/perm:output:0*
T0*+
_output_shapes
:���������dc
lstm_66/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    �
dense_198/MatMul/ReadVariableOpReadVariableOp(dense_198_matmul_readvariableop_resource*
_output_shapes

:d2*
dtype0�
dense_198/MatMulMatMul lstm_66/strided_slice_3:output:0'dense_198/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2�
 dense_198/BiasAdd/ReadVariableOpReadVariableOp)dense_198_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0�
dense_198/BiasAddBiasAdddense_198/MatMul:product:0(dense_198/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2d
dense_198/ReluReludense_198/BiasAdd:output:0*
T0*'
_output_shapes
:���������2�
dense_199/MatMul/ReadVariableOpReadVariableOp(dense_199_matmul_readvariableop_resource*
_output_shapes

:22*
dtype0�
dense_199/MatMulMatMuldense_198/Relu:activations:0'dense_199/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2�
 dense_199/BiasAdd/ReadVariableOpReadVariableOp)dense_199_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0�
dense_199/BiasAddBiasAdddense_199/MatMul:product:0(dense_199/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2d
dense_199/ReluReludense_199/BiasAdd:output:0*
T0*'
_output_shapes
:���������2�
dense_200/MatMul/ReadVariableOpReadVariableOp(dense_200_matmul_readvariableop_resource*
_output_shapes

:2*
dtype0�
dense_200/MatMulMatMuldense_199/Relu:activations:0'dense_200/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
 dense_200/BiasAdd/ReadVariableOpReadVariableOp)dense_200_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_200/BiasAddBiasAdddense_200/MatMul:product:0(dense_200/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������i
IdentityIdentitydense_200/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp!^dense_198/BiasAdd/ReadVariableOp ^dense_198/MatMul/ReadVariableOp!^dense_199/BiasAdd/ReadVariableOp ^dense_199/MatMul/ReadVariableOp!^dense_200/BiasAdd/ReadVariableOp ^dense_200/MatMul/ReadVariableOp,^lstm_66/lstm_cell_66/BiasAdd/ReadVariableOp+^lstm_66/lstm_cell_66/MatMul/ReadVariableOp-^lstm_66/lstm_cell_66/MatMul_1/ReadVariableOp^lstm_66/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):���������: : : : : : : : : 2D
 dense_198/BiasAdd/ReadVariableOp dense_198/BiasAdd/ReadVariableOp2B
dense_198/MatMul/ReadVariableOpdense_198/MatMul/ReadVariableOp2D
 dense_199/BiasAdd/ReadVariableOp dense_199/BiasAdd/ReadVariableOp2B
dense_199/MatMul/ReadVariableOpdense_199/MatMul/ReadVariableOp2D
 dense_200/BiasAdd/ReadVariableOp dense_200/BiasAdd/ReadVariableOp2B
dense_200/MatMul/ReadVariableOpdense_200/MatMul/ReadVariableOp2Z
+lstm_66/lstm_cell_66/BiasAdd/ReadVariableOp+lstm_66/lstm_cell_66/BiasAdd/ReadVariableOp2X
*lstm_66/lstm_cell_66/MatMul/ReadVariableOp*lstm_66/lstm_cell_66/MatMul/ReadVariableOp2\
,lstm_66/lstm_cell_66/MatMul_1/ReadVariableOp,lstm_66/lstm_cell_66/MatMul_1/ReadVariableOp2
lstm_66/whilelstm_66/while:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
while_cond_437463124
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_17
3while_while_cond_437463124___redundant_placeholder07
3while_while_cond_437463124___redundant_placeholder17
3while_while_cond_437463124___redundant_placeholder27
3while_while_cond_437463124___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
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
@: : : : :���������d:���������d: ::::: 
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
:���������d:-)
'
_output_shapes
:���������d:

_output_shapes
: :

_output_shapes
:
�
�
while_cond_437464289
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_17
3while_while_cond_437464289___redundant_placeholder07
3while_while_cond_437464289___redundant_placeholder17
3while_while_cond_437464289___redundant_placeholder27
3while_while_cond_437464289___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
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
@: : : : :���������d:���������d: ::::: 
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
:���������d:-)
'
_output_shapes
:���������d:

_output_shapes
: :

_output_shapes
:
�

�
1__inference_sequential_66_layer_call_fn_437463435
lstm_66_input
unknown:	�
	unknown_0:	d�
	unknown_1:	�
	unknown_2:d2
	unknown_3:2
	unknown_4:22
	unknown_5:2
	unknown_6:2
	unknown_7:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalllstm_66_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*+
_read_only_resource_inputs
		*-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_sequential_66_layer_call_and_return_conditional_losses_437463414o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):���������: : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
+
_output_shapes
:���������
'
_user_specified_namelstm_66_input
�
�
+__inference_lstm_66_layer_call_fn_437464219

inputs
unknown:	�
	unknown_0:	d�
	unknown_1:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_lstm_66_layer_call_and_return_conditional_losses_437463355o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�K
�
F__inference_lstm_66_layer_call_and_return_conditional_losses_437464665

inputs>
+lstm_cell_66_matmul_readvariableop_resource:	�@
-lstm_cell_66_matmul_1_readvariableop_resource:	d�;
,lstm_cell_66_biasadd_readvariableop_resource:	�
identity��#lstm_cell_66/BiasAdd/ReadVariableOp�"lstm_cell_66/MatMul/ReadVariableOp�$lstm_cell_66/MatMul_1/ReadVariableOp�while;
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
valueB:�
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
value	B :ds
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
:���������dR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :dw
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
:���������dc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:���������D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
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
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
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
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
"lstm_cell_66/MatMul/ReadVariableOpReadVariableOp+lstm_cell_66_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
lstm_cell_66/MatMulMatMulstrided_slice_2:output:0*lstm_cell_66/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
$lstm_cell_66/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_66_matmul_1_readvariableop_resource*
_output_shapes
:	d�*
dtype0�
lstm_cell_66/MatMul_1MatMulzeros:output:0,lstm_cell_66/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
lstm_cell_66/addAddV2lstm_cell_66/MatMul:product:0lstm_cell_66/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
#lstm_cell_66/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_66_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
lstm_cell_66/BiasAddBiasAddlstm_cell_66/add:z:0+lstm_cell_66/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������^
lstm_cell_66/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_cell_66/splitSplit%lstm_cell_66/split/split_dim:output:0lstm_cell_66/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������d:���������d:���������d:���������d*
	num_splitn
lstm_cell_66/SigmoidSigmoidlstm_cell_66/split:output:0*
T0*'
_output_shapes
:���������dp
lstm_cell_66/Sigmoid_1Sigmoidlstm_cell_66/split:output:1*
T0*'
_output_shapes
:���������dw
lstm_cell_66/mulMullstm_cell_66/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������dh
lstm_cell_66/ReluRelulstm_cell_66/split:output:2*
T0*'
_output_shapes
:���������d�
lstm_cell_66/mul_1Mullstm_cell_66/Sigmoid:y:0lstm_cell_66/Relu:activations:0*
T0*'
_output_shapes
:���������d{
lstm_cell_66/add_1AddV2lstm_cell_66/mul:z:0lstm_cell_66/mul_1:z:0*
T0*'
_output_shapes
:���������dp
lstm_cell_66/Sigmoid_2Sigmoidlstm_cell_66/split:output:3*
T0*'
_output_shapes
:���������de
lstm_cell_66/Relu_1Relulstm_cell_66/add_1:z:0*
T0*'
_output_shapes
:���������d�
lstm_cell_66/mul_2Mullstm_cell_66/Sigmoid_2:y:0!lstm_cell_66/Relu_1:activations:0*
T0*'
_output_shapes
:���������dn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
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
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_66_matmul_readvariableop_resource-lstm_cell_66_matmul_1_readvariableop_resource,lstm_cell_66_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������d:���������d: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( * 
bodyR
while_body_437464580* 
condR
while_cond_437464579*K
output_shapes:
8: : : : :���������d:���������d: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������d*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������d*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:���������d[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:���������d�
NoOpNoOp$^lstm_cell_66/BiasAdd/ReadVariableOp#^lstm_cell_66/MatMul/ReadVariableOp%^lstm_cell_66/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������: : : 2J
#lstm_cell_66/BiasAdd/ReadVariableOp#lstm_cell_66/BiasAdd/ReadVariableOp2H
"lstm_cell_66/MatMul/ReadVariableOp"lstm_cell_66/MatMul/ReadVariableOp2L
$lstm_cell_66/MatMul_1/ReadVariableOp$lstm_cell_66/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�K
�
F__inference_lstm_66_layer_call_and_return_conditional_losses_437464375
inputs_0>
+lstm_cell_66_matmul_readvariableop_resource:	�@
-lstm_cell_66_matmul_1_readvariableop_resource:	d�;
,lstm_cell_66_biasadd_readvariableop_resource:	�
identity��#lstm_cell_66/BiasAdd/ReadVariableOp�"lstm_cell_66/MatMul/ReadVariableOp�$lstm_cell_66/MatMul_1/ReadVariableOp�while=
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
valueB:�
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
value	B :ds
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
:���������dR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :dw
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
:���������dc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :������������������D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
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
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
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
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
"lstm_cell_66/MatMul/ReadVariableOpReadVariableOp+lstm_cell_66_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
lstm_cell_66/MatMulMatMulstrided_slice_2:output:0*lstm_cell_66/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
$lstm_cell_66/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_66_matmul_1_readvariableop_resource*
_output_shapes
:	d�*
dtype0�
lstm_cell_66/MatMul_1MatMulzeros:output:0,lstm_cell_66/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
lstm_cell_66/addAddV2lstm_cell_66/MatMul:product:0lstm_cell_66/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
#lstm_cell_66/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_66_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
lstm_cell_66/BiasAddBiasAddlstm_cell_66/add:z:0+lstm_cell_66/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������^
lstm_cell_66/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_cell_66/splitSplit%lstm_cell_66/split/split_dim:output:0lstm_cell_66/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������d:���������d:���������d:���������d*
	num_splitn
lstm_cell_66/SigmoidSigmoidlstm_cell_66/split:output:0*
T0*'
_output_shapes
:���������dp
lstm_cell_66/Sigmoid_1Sigmoidlstm_cell_66/split:output:1*
T0*'
_output_shapes
:���������dw
lstm_cell_66/mulMullstm_cell_66/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������dh
lstm_cell_66/ReluRelulstm_cell_66/split:output:2*
T0*'
_output_shapes
:���������d�
lstm_cell_66/mul_1Mullstm_cell_66/Sigmoid:y:0lstm_cell_66/Relu:activations:0*
T0*'
_output_shapes
:���������d{
lstm_cell_66/add_1AddV2lstm_cell_66/mul:z:0lstm_cell_66/mul_1:z:0*
T0*'
_output_shapes
:���������dp
lstm_cell_66/Sigmoid_2Sigmoidlstm_cell_66/split:output:3*
T0*'
_output_shapes
:���������de
lstm_cell_66/Relu_1Relulstm_cell_66/add_1:z:0*
T0*'
_output_shapes
:���������d�
lstm_cell_66/mul_2Mullstm_cell_66/Sigmoid_2:y:0!lstm_cell_66/Relu_1:activations:0*
T0*'
_output_shapes
:���������dn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
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
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_66_matmul_readvariableop_resource-lstm_cell_66_matmul_1_readvariableop_resource,lstm_cell_66_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������d:���������d: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( * 
bodyR
while_body_437464290* 
condR
while_cond_437464289*K
output_shapes:
8: : : : :���������d:���������d: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������d*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������d*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:���������d[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:���������d�
NoOpNoOp$^lstm_cell_66/BiasAdd/ReadVariableOp#^lstm_cell_66/MatMul/ReadVariableOp%^lstm_cell_66/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 2J
#lstm_cell_66/BiasAdd/ReadVariableOp#lstm_cell_66/BiasAdd/ReadVariableOp2H
"lstm_cell_66/MatMul/ReadVariableOp"lstm_cell_66/MatMul/ReadVariableOp2L
$lstm_cell_66/MatMul_1/ReadVariableOp$lstm_cell_66/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :������������������
"
_user_specified_name
inputs/0
�

�
lstm_66_while_cond_437464080,
(lstm_66_while_lstm_66_while_loop_counter2
.lstm_66_while_lstm_66_while_maximum_iterations
lstm_66_while_placeholder
lstm_66_while_placeholder_1
lstm_66_while_placeholder_2
lstm_66_while_placeholder_3.
*lstm_66_while_less_lstm_66_strided_slice_1G
Clstm_66_while_lstm_66_while_cond_437464080___redundant_placeholder0G
Clstm_66_while_lstm_66_while_cond_437464080___redundant_placeholder1G
Clstm_66_while_lstm_66_while_cond_437464080___redundant_placeholder2G
Clstm_66_while_lstm_66_while_cond_437464080___redundant_placeholder3
lstm_66_while_identity
�
lstm_66/while/LessLesslstm_66_while_placeholder*lstm_66_while_less_lstm_66_strided_slice_1*
T0*
_output_shapes
: [
lstm_66/while/IdentityIdentitylstm_66/while/Less:z:0*
T0
*
_output_shapes
: "9
lstm_66_while_identitylstm_66/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :���������d:���������d: ::::: 
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
:���������d:-)
'
_output_shapes
:���������d:

_output_shapes
: :

_output_shapes
:
�

�
1__inference_sequential_66_layer_call_fn_437463856

inputs
unknown:	�
	unknown_0:	d�
	unknown_1:	�
	unknown_2:d2
	unknown_3:2
	unknown_4:22
	unknown_5:2
	unknown_6:2
	unknown_7:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*+
_read_only_resource_inputs
		*-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_sequential_66_layer_call_and_return_conditional_losses_437463683o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):���������: : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
+__inference_lstm_66_layer_call_fn_437464208
inputs_0
unknown:	�
	unknown_0:	d�
	unknown_1:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_lstm_66_layer_call_and_return_conditional_losses_437463195o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :������������������
"
_user_specified_name
inputs/0
�
�
while_cond_437464724
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_17
3while_while_cond_437464724___redundant_placeholder07
3while_while_cond_437464724___redundant_placeholder17
3while_while_cond_437464724___redundant_placeholder27
3while_while_cond_437464724___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
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
@: : : : :���������d:���������d: ::::: 
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
:���������d:-)
'
_output_shapes
:���������d:

_output_shapes
: :

_output_shapes
:
�
�
+__inference_lstm_66_layer_call_fn_437464230

inputs
unknown:	�
	unknown_0:	d�
	unknown_1:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_lstm_66_layer_call_and_return_conditional_losses_437463623o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
L__inference_sequential_66_layer_call_and_return_conditional_losses_437463683

inputs$
lstm_66_437463660:	�$
lstm_66_437463662:	d� 
lstm_66_437463664:	�%
dense_198_437463667:d2!
dense_198_437463669:2%
dense_199_437463672:22!
dense_199_437463674:2%
dense_200_437463677:2!
dense_200_437463679:
identity��!dense_198/StatefulPartitionedCall�!dense_199/StatefulPartitionedCall�!dense_200/StatefulPartitionedCall�lstm_66/StatefulPartitionedCall�
lstm_66/StatefulPartitionedCallStatefulPartitionedCallinputslstm_66_437463660lstm_66_437463662lstm_66_437463664*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_lstm_66_layer_call_and_return_conditional_losses_437463623�
!dense_198/StatefulPartitionedCallStatefulPartitionedCall(lstm_66/StatefulPartitionedCall:output:0dense_198_437463667dense_198_437463669*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_dense_198_layer_call_and_return_conditional_losses_437463374�
!dense_199/StatefulPartitionedCallStatefulPartitionedCall*dense_198/StatefulPartitionedCall:output:0dense_199_437463672dense_199_437463674*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_dense_199_layer_call_and_return_conditional_losses_437463391�
!dense_200/StatefulPartitionedCallStatefulPartitionedCall*dense_199/StatefulPartitionedCall:output:0dense_200_437463677dense_200_437463679*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_dense_200_layer_call_and_return_conditional_losses_437463407y
IdentityIdentity*dense_200/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp"^dense_198/StatefulPartitionedCall"^dense_199/StatefulPartitionedCall"^dense_200/StatefulPartitionedCall ^lstm_66/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):���������: : : : : : : : : 2F
!dense_198/StatefulPartitionedCall!dense_198/StatefulPartitionedCall2F
!dense_199/StatefulPartitionedCall!dense_199/StatefulPartitionedCall2F
!dense_200/StatefulPartitionedCall!dense_200/StatefulPartitionedCall2B
lstm_66/StatefulPartitionedCalllstm_66/StatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
1__inference_sequential_66_layer_call_fn_437463833

inputs
unknown:	�
	unknown_0:	d�
	unknown_1:	�
	unknown_2:d2
	unknown_3:2
	unknown_4:22
	unknown_5:2
	unknown_6:2
	unknown_7:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*+
_read_only_resource_inputs
		*-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_sequential_66_layer_call_and_return_conditional_losses_437463414o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):���������: : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
H__inference_dense_198_layer_call_and_return_conditional_losses_437464830

inputs0
matmul_readvariableop_resource:d2-
biasadd_readvariableop_resource:2
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d2*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������2a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������2w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�9
�
while_body_437464725
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_66_matmul_readvariableop_resource_0:	�H
5while_lstm_cell_66_matmul_1_readvariableop_resource_0:	d�C
4while_lstm_cell_66_biasadd_readvariableop_resource_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_66_matmul_readvariableop_resource:	�F
3while_lstm_cell_66_matmul_1_readvariableop_resource:	d�A
2while_lstm_cell_66_biasadd_readvariableop_resource:	���)while/lstm_cell_66/BiasAdd/ReadVariableOp�(while/lstm_cell_66/MatMul/ReadVariableOp�*while/lstm_cell_66/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
(while/lstm_cell_66/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_66_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
while/lstm_cell_66/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_66/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
*while/lstm_cell_66/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_66_matmul_1_readvariableop_resource_0*
_output_shapes
:	d�*
dtype0�
while/lstm_cell_66/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_66/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
while/lstm_cell_66/addAddV2#while/lstm_cell_66/MatMul:product:0%while/lstm_cell_66/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
)while/lstm_cell_66/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_66_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype0�
while/lstm_cell_66/BiasAddBiasAddwhile/lstm_cell_66/add:z:01while/lstm_cell_66/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������d
"while/lstm_cell_66/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/lstm_cell_66/splitSplit+while/lstm_cell_66/split/split_dim:output:0#while/lstm_cell_66/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������d:���������d:���������d:���������d*
	num_splitz
while/lstm_cell_66/SigmoidSigmoid!while/lstm_cell_66/split:output:0*
T0*'
_output_shapes
:���������d|
while/lstm_cell_66/Sigmoid_1Sigmoid!while/lstm_cell_66/split:output:1*
T0*'
_output_shapes
:���������d�
while/lstm_cell_66/mulMul while/lstm_cell_66/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������dt
while/lstm_cell_66/ReluRelu!while/lstm_cell_66/split:output:2*
T0*'
_output_shapes
:���������d�
while/lstm_cell_66/mul_1Mulwhile/lstm_cell_66/Sigmoid:y:0%while/lstm_cell_66/Relu:activations:0*
T0*'
_output_shapes
:���������d�
while/lstm_cell_66/add_1AddV2while/lstm_cell_66/mul:z:0while/lstm_cell_66/mul_1:z:0*
T0*'
_output_shapes
:���������d|
while/lstm_cell_66/Sigmoid_2Sigmoid!while/lstm_cell_66/split:output:3*
T0*'
_output_shapes
:���������dq
while/lstm_cell_66/Relu_1Reluwhile/lstm_cell_66/add_1:z:0*
T0*'
_output_shapes
:���������d�
while/lstm_cell_66/mul_2Mul while/lstm_cell_66/Sigmoid_2:y:0'while/lstm_cell_66/Relu_1:activations:0*
T0*'
_output_shapes
:���������dr
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : �
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell_66/mul_2:z:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: y
while/Identity_4Identitywhile/lstm_cell_66/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:���������dy
while/Identity_5Identitywhile/lstm_cell_66/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:���������d�

while/NoOpNoOp*^while/lstm_cell_66/BiasAdd/ReadVariableOp)^while/lstm_cell_66/MatMul/ReadVariableOp+^while/lstm_cell_66/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_66_biasadd_readvariableop_resource4while_lstm_cell_66_biasadd_readvariableop_resource_0"l
3while_lstm_cell_66_matmul_1_readvariableop_resource5while_lstm_cell_66_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_66_matmul_readvariableop_resource3while_lstm_cell_66_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������d:���������d: : : : : 2V
)while/lstm_cell_66/BiasAdd/ReadVariableOp)while/lstm_cell_66/BiasAdd/ReadVariableOp2T
(while/lstm_cell_66/MatMul/ReadVariableOp(while/lstm_cell_66/MatMul/ReadVariableOp2X
*while/lstm_cell_66/MatMul_1/ReadVariableOp*while/lstm_cell_66/MatMul_1/ReadVariableOp: 
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
:���������d:-)
'
_output_shapes
:���������d:

_output_shapes
: :

_output_shapes
: 
�
�
K__inference_lstm_cell_66_layer_call_and_return_conditional_losses_437462917

inputs

states
states_11
matmul_readvariableop_resource:	�3
 matmul_1_readvariableop_resource:	d�.
biasadd_readvariableop_resource:	�
identity

identity_1

identity_2��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������y
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	d�*
dtype0n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������e
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:���������d:���������d:���������d:���������d*
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:���������dV
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:���������dU
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:���������dN
ReluRelusplit:output:2*
T0*'
_output_shapes
:���������d_
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:���������dT
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:���������dV
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:���������dK
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:���������dc
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:���������dX
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:���������dZ

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:���������dZ

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:���������d�
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������:���������d:���������d: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������d
 
_user_specified_namestates:OK
'
_output_shapes
:���������d
 
_user_specified_namestates
�9
�
while_body_437464290
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_66_matmul_readvariableop_resource_0:	�H
5while_lstm_cell_66_matmul_1_readvariableop_resource_0:	d�C
4while_lstm_cell_66_biasadd_readvariableop_resource_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_66_matmul_readvariableop_resource:	�F
3while_lstm_cell_66_matmul_1_readvariableop_resource:	d�A
2while_lstm_cell_66_biasadd_readvariableop_resource:	���)while/lstm_cell_66/BiasAdd/ReadVariableOp�(while/lstm_cell_66/MatMul/ReadVariableOp�*while/lstm_cell_66/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
(while/lstm_cell_66/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_66_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
while/lstm_cell_66/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_66/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
*while/lstm_cell_66/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_66_matmul_1_readvariableop_resource_0*
_output_shapes
:	d�*
dtype0�
while/lstm_cell_66/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_66/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
while/lstm_cell_66/addAddV2#while/lstm_cell_66/MatMul:product:0%while/lstm_cell_66/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
)while/lstm_cell_66/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_66_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype0�
while/lstm_cell_66/BiasAddBiasAddwhile/lstm_cell_66/add:z:01while/lstm_cell_66/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������d
"while/lstm_cell_66/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/lstm_cell_66/splitSplit+while/lstm_cell_66/split/split_dim:output:0#while/lstm_cell_66/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������d:���������d:���������d:���������d*
	num_splitz
while/lstm_cell_66/SigmoidSigmoid!while/lstm_cell_66/split:output:0*
T0*'
_output_shapes
:���������d|
while/lstm_cell_66/Sigmoid_1Sigmoid!while/lstm_cell_66/split:output:1*
T0*'
_output_shapes
:���������d�
while/lstm_cell_66/mulMul while/lstm_cell_66/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������dt
while/lstm_cell_66/ReluRelu!while/lstm_cell_66/split:output:2*
T0*'
_output_shapes
:���������d�
while/lstm_cell_66/mul_1Mulwhile/lstm_cell_66/Sigmoid:y:0%while/lstm_cell_66/Relu:activations:0*
T0*'
_output_shapes
:���������d�
while/lstm_cell_66/add_1AddV2while/lstm_cell_66/mul:z:0while/lstm_cell_66/mul_1:z:0*
T0*'
_output_shapes
:���������d|
while/lstm_cell_66/Sigmoid_2Sigmoid!while/lstm_cell_66/split:output:3*
T0*'
_output_shapes
:���������dq
while/lstm_cell_66/Relu_1Reluwhile/lstm_cell_66/add_1:z:0*
T0*'
_output_shapes
:���������d�
while/lstm_cell_66/mul_2Mul while/lstm_cell_66/Sigmoid_2:y:0'while/lstm_cell_66/Relu_1:activations:0*
T0*'
_output_shapes
:���������dr
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : �
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell_66/mul_2:z:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: y
while/Identity_4Identitywhile/lstm_cell_66/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:���������dy
while/Identity_5Identitywhile/lstm_cell_66/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:���������d�

while/NoOpNoOp*^while/lstm_cell_66/BiasAdd/ReadVariableOp)^while/lstm_cell_66/MatMul/ReadVariableOp+^while/lstm_cell_66/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_66_biasadd_readvariableop_resource4while_lstm_cell_66_biasadd_readvariableop_resource_0"l
3while_lstm_cell_66_matmul_1_readvariableop_resource5while_lstm_cell_66_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_66_matmul_readvariableop_resource3while_lstm_cell_66_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������d:���������d: : : : : 2V
)while/lstm_cell_66/BiasAdd/ReadVariableOp)while/lstm_cell_66/BiasAdd/ReadVariableOp2T
(while/lstm_cell_66/MatMul/ReadVariableOp(while/lstm_cell_66/MatMul/ReadVariableOp2X
*while/lstm_cell_66/MatMul_1/ReadVariableOp*while/lstm_cell_66/MatMul_1/ReadVariableOp: 
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
:���������d:-)
'
_output_shapes
:���������d:

_output_shapes
: :

_output_shapes
: 
�	
�
H__inference_dense_200_layer_call_and_return_conditional_losses_437464869

inputs0
matmul_readvariableop_resource:2-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:2*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������2: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������2
 
_user_specified_nameinputs
�9
�
F__inference_lstm_66_layer_call_and_return_conditional_losses_437463195

inputs)
lstm_cell_66_437463111:	�)
lstm_cell_66_437463113:	d�%
lstm_cell_66_437463115:	�
identity��$lstm_cell_66/StatefulPartitionedCall�while;
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
valueB:�
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
value	B :ds
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
:���������dR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :dw
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
:���������dc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :������������������D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
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
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
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
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
$lstm_cell_66/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_66_437463111lstm_cell_66_437463113lstm_cell_66_437463115*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:���������d:���������d:���������d*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_lstm_cell_66_layer_call_and_return_conditional_losses_437463065n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
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
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_66_437463111lstm_cell_66_437463113lstm_cell_66_437463115*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������d:���������d: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( * 
bodyR
while_body_437463125* 
condR
while_cond_437463124*K
output_shapes:
8: : : : :���������d:���������d: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������d*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������d*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:���������d[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:���������du
NoOpNoOp%^lstm_cell_66/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 2L
$lstm_cell_66/StatefulPartitionedCall$lstm_cell_66/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs
�K
�
F__inference_lstm_66_layer_call_and_return_conditional_losses_437463623

inputs>
+lstm_cell_66_matmul_readvariableop_resource:	�@
-lstm_cell_66_matmul_1_readvariableop_resource:	d�;
,lstm_cell_66_biasadd_readvariableop_resource:	�
identity��#lstm_cell_66/BiasAdd/ReadVariableOp�"lstm_cell_66/MatMul/ReadVariableOp�$lstm_cell_66/MatMul_1/ReadVariableOp�while;
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
valueB:�
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
value	B :ds
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
:���������dR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :dw
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
:���������dc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:���������D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
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
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
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
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
"lstm_cell_66/MatMul/ReadVariableOpReadVariableOp+lstm_cell_66_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
lstm_cell_66/MatMulMatMulstrided_slice_2:output:0*lstm_cell_66/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
$lstm_cell_66/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_66_matmul_1_readvariableop_resource*
_output_shapes
:	d�*
dtype0�
lstm_cell_66/MatMul_1MatMulzeros:output:0,lstm_cell_66/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
lstm_cell_66/addAddV2lstm_cell_66/MatMul:product:0lstm_cell_66/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
#lstm_cell_66/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_66_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
lstm_cell_66/BiasAddBiasAddlstm_cell_66/add:z:0+lstm_cell_66/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������^
lstm_cell_66/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_cell_66/splitSplit%lstm_cell_66/split/split_dim:output:0lstm_cell_66/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������d:���������d:���������d:���������d*
	num_splitn
lstm_cell_66/SigmoidSigmoidlstm_cell_66/split:output:0*
T0*'
_output_shapes
:���������dp
lstm_cell_66/Sigmoid_1Sigmoidlstm_cell_66/split:output:1*
T0*'
_output_shapes
:���������dw
lstm_cell_66/mulMullstm_cell_66/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������dh
lstm_cell_66/ReluRelulstm_cell_66/split:output:2*
T0*'
_output_shapes
:���������d�
lstm_cell_66/mul_1Mullstm_cell_66/Sigmoid:y:0lstm_cell_66/Relu:activations:0*
T0*'
_output_shapes
:���������d{
lstm_cell_66/add_1AddV2lstm_cell_66/mul:z:0lstm_cell_66/mul_1:z:0*
T0*'
_output_shapes
:���������dp
lstm_cell_66/Sigmoid_2Sigmoidlstm_cell_66/split:output:3*
T0*'
_output_shapes
:���������de
lstm_cell_66/Relu_1Relulstm_cell_66/add_1:z:0*
T0*'
_output_shapes
:���������d�
lstm_cell_66/mul_2Mullstm_cell_66/Sigmoid_2:y:0!lstm_cell_66/Relu_1:activations:0*
T0*'
_output_shapes
:���������dn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
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
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_66_matmul_readvariableop_resource-lstm_cell_66_matmul_1_readvariableop_resource,lstm_cell_66_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������d:���������d: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( * 
bodyR
while_body_437463538* 
condR
while_cond_437463537*K
output_shapes:
8: : : : :���������d:���������d: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������d*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������d*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:���������d[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:���������d�
NoOpNoOp$^lstm_cell_66/BiasAdd/ReadVariableOp#^lstm_cell_66/MatMul/ReadVariableOp%^lstm_cell_66/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������: : : 2J
#lstm_cell_66/BiasAdd/ReadVariableOp#lstm_cell_66/BiasAdd/ReadVariableOp2H
"lstm_cell_66/MatMul/ReadVariableOp"lstm_cell_66/MatMul/ReadVariableOp2L
$lstm_cell_66/MatMul_1/ReadVariableOp$lstm_cell_66/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�m
�
L__inference_sequential_66_layer_call_and_return_conditional_losses_437464021

inputsF
3lstm_66_lstm_cell_66_matmul_readvariableop_resource:	�H
5lstm_66_lstm_cell_66_matmul_1_readvariableop_resource:	d�C
4lstm_66_lstm_cell_66_biasadd_readvariableop_resource:	�:
(dense_198_matmul_readvariableop_resource:d27
)dense_198_biasadd_readvariableop_resource:2:
(dense_199_matmul_readvariableop_resource:227
)dense_199_biasadd_readvariableop_resource:2:
(dense_200_matmul_readvariableop_resource:27
)dense_200_biasadd_readvariableop_resource:
identity�� dense_198/BiasAdd/ReadVariableOp�dense_198/MatMul/ReadVariableOp� dense_199/BiasAdd/ReadVariableOp�dense_199/MatMul/ReadVariableOp� dense_200/BiasAdd/ReadVariableOp�dense_200/MatMul/ReadVariableOp�+lstm_66/lstm_cell_66/BiasAdd/ReadVariableOp�*lstm_66/lstm_cell_66/MatMul/ReadVariableOp�,lstm_66/lstm_cell_66/MatMul_1/ReadVariableOp�lstm_66/whileC
lstm_66/ShapeShapeinputs*
T0*
_output_shapes
:e
lstm_66/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
lstm_66/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
lstm_66/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
lstm_66/strided_sliceStridedSlicelstm_66/Shape:output:0$lstm_66/strided_slice/stack:output:0&lstm_66/strided_slice/stack_1:output:0&lstm_66/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
lstm_66/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d�
lstm_66/zeros/packedPacklstm_66/strided_slice:output:0lstm_66/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:X
lstm_66/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
lstm_66/zerosFilllstm_66/zeros/packed:output:0lstm_66/zeros/Const:output:0*
T0*'
_output_shapes
:���������dZ
lstm_66/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d�
lstm_66/zeros_1/packedPacklstm_66/strided_slice:output:0!lstm_66/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:Z
lstm_66/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
lstm_66/zeros_1Filllstm_66/zeros_1/packed:output:0lstm_66/zeros_1/Const:output:0*
T0*'
_output_shapes
:���������dk
lstm_66/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          }
lstm_66/transpose	Transposeinputslstm_66/transpose/perm:output:0*
T0*+
_output_shapes
:���������T
lstm_66/Shape_1Shapelstm_66/transpose:y:0*
T0*
_output_shapes
:g
lstm_66/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: i
lstm_66/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
lstm_66/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
lstm_66/strided_slice_1StridedSlicelstm_66/Shape_1:output:0&lstm_66/strided_slice_1/stack:output:0(lstm_66/strided_slice_1/stack_1:output:0(lstm_66/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskn
#lstm_66/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
lstm_66/TensorArrayV2TensorListReserve,lstm_66/TensorArrayV2/element_shape:output:0 lstm_66/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
=lstm_66/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
/lstm_66/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_66/transpose:y:0Flstm_66/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���g
lstm_66/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: i
lstm_66/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
lstm_66/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
lstm_66/strided_slice_2StridedSlicelstm_66/transpose:y:0&lstm_66/strided_slice_2/stack:output:0(lstm_66/strided_slice_2/stack_1:output:0(lstm_66/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
*lstm_66/lstm_cell_66/MatMul/ReadVariableOpReadVariableOp3lstm_66_lstm_cell_66_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
lstm_66/lstm_cell_66/MatMulMatMul lstm_66/strided_slice_2:output:02lstm_66/lstm_cell_66/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
,lstm_66/lstm_cell_66/MatMul_1/ReadVariableOpReadVariableOp5lstm_66_lstm_cell_66_matmul_1_readvariableop_resource*
_output_shapes
:	d�*
dtype0�
lstm_66/lstm_cell_66/MatMul_1MatMullstm_66/zeros:output:04lstm_66/lstm_cell_66/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
lstm_66/lstm_cell_66/addAddV2%lstm_66/lstm_cell_66/MatMul:product:0'lstm_66/lstm_cell_66/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
+lstm_66/lstm_cell_66/BiasAdd/ReadVariableOpReadVariableOp4lstm_66_lstm_cell_66_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
lstm_66/lstm_cell_66/BiasAddBiasAddlstm_66/lstm_cell_66/add:z:03lstm_66/lstm_cell_66/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������f
$lstm_66/lstm_cell_66/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_66/lstm_cell_66/splitSplit-lstm_66/lstm_cell_66/split/split_dim:output:0%lstm_66/lstm_cell_66/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������d:���������d:���������d:���������d*
	num_split~
lstm_66/lstm_cell_66/SigmoidSigmoid#lstm_66/lstm_cell_66/split:output:0*
T0*'
_output_shapes
:���������d�
lstm_66/lstm_cell_66/Sigmoid_1Sigmoid#lstm_66/lstm_cell_66/split:output:1*
T0*'
_output_shapes
:���������d�
lstm_66/lstm_cell_66/mulMul"lstm_66/lstm_cell_66/Sigmoid_1:y:0lstm_66/zeros_1:output:0*
T0*'
_output_shapes
:���������dx
lstm_66/lstm_cell_66/ReluRelu#lstm_66/lstm_cell_66/split:output:2*
T0*'
_output_shapes
:���������d�
lstm_66/lstm_cell_66/mul_1Mul lstm_66/lstm_cell_66/Sigmoid:y:0'lstm_66/lstm_cell_66/Relu:activations:0*
T0*'
_output_shapes
:���������d�
lstm_66/lstm_cell_66/add_1AddV2lstm_66/lstm_cell_66/mul:z:0lstm_66/lstm_cell_66/mul_1:z:0*
T0*'
_output_shapes
:���������d�
lstm_66/lstm_cell_66/Sigmoid_2Sigmoid#lstm_66/lstm_cell_66/split:output:3*
T0*'
_output_shapes
:���������du
lstm_66/lstm_cell_66/Relu_1Relulstm_66/lstm_cell_66/add_1:z:0*
T0*'
_output_shapes
:���������d�
lstm_66/lstm_cell_66/mul_2Mul"lstm_66/lstm_cell_66/Sigmoid_2:y:0)lstm_66/lstm_cell_66/Relu_1:activations:0*
T0*'
_output_shapes
:���������dv
%lstm_66/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   f
$lstm_66/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_66/TensorArrayV2_1TensorListReserve.lstm_66/TensorArrayV2_1/element_shape:output:0-lstm_66/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���N
lstm_66/timeConst*
_output_shapes
: *
dtype0*
value	B : k
 lstm_66/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������\
lstm_66/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
lstm_66/whileWhile#lstm_66/while/loop_counter:output:0)lstm_66/while/maximum_iterations:output:0lstm_66/time:output:0 lstm_66/TensorArrayV2_1:handle:0lstm_66/zeros:output:0lstm_66/zeros_1:output:0 lstm_66/strided_slice_1:output:0?lstm_66/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_66_lstm_cell_66_matmul_readvariableop_resource5lstm_66_lstm_cell_66_matmul_1_readvariableop_resource4lstm_66_lstm_cell_66_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������d:���������d: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *(
body R
lstm_66_while_body_437463916*(
cond R
lstm_66_while_cond_437463915*K
output_shapes:
8: : : : :���������d:���������d: : : : : *
parallel_iterations �
8lstm_66/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   �
*lstm_66/TensorArrayV2Stack/TensorListStackTensorListStacklstm_66/while:output:3Alstm_66/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������d*
element_dtype0*
num_elementsp
lstm_66/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������i
lstm_66/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: i
lstm_66/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
lstm_66/strided_slice_3StridedSlice3lstm_66/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_66/strided_slice_3/stack:output:0(lstm_66/strided_slice_3/stack_1:output:0(lstm_66/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������d*
shrink_axis_maskm
lstm_66/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
lstm_66/transpose_1	Transpose3lstm_66/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_66/transpose_1/perm:output:0*
T0*+
_output_shapes
:���������dc
lstm_66/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    �
dense_198/MatMul/ReadVariableOpReadVariableOp(dense_198_matmul_readvariableop_resource*
_output_shapes

:d2*
dtype0�
dense_198/MatMulMatMul lstm_66/strided_slice_3:output:0'dense_198/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2�
 dense_198/BiasAdd/ReadVariableOpReadVariableOp)dense_198_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0�
dense_198/BiasAddBiasAdddense_198/MatMul:product:0(dense_198/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2d
dense_198/ReluReludense_198/BiasAdd:output:0*
T0*'
_output_shapes
:���������2�
dense_199/MatMul/ReadVariableOpReadVariableOp(dense_199_matmul_readvariableop_resource*
_output_shapes

:22*
dtype0�
dense_199/MatMulMatMuldense_198/Relu:activations:0'dense_199/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2�
 dense_199/BiasAdd/ReadVariableOpReadVariableOp)dense_199_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0�
dense_199/BiasAddBiasAdddense_199/MatMul:product:0(dense_199/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2d
dense_199/ReluReludense_199/BiasAdd:output:0*
T0*'
_output_shapes
:���������2�
dense_200/MatMul/ReadVariableOpReadVariableOp(dense_200_matmul_readvariableop_resource*
_output_shapes

:2*
dtype0�
dense_200/MatMulMatMuldense_199/Relu:activations:0'dense_200/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
 dense_200/BiasAdd/ReadVariableOpReadVariableOp)dense_200_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_200/BiasAddBiasAdddense_200/MatMul:product:0(dense_200/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������i
IdentityIdentitydense_200/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp!^dense_198/BiasAdd/ReadVariableOp ^dense_198/MatMul/ReadVariableOp!^dense_199/BiasAdd/ReadVariableOp ^dense_199/MatMul/ReadVariableOp!^dense_200/BiasAdd/ReadVariableOp ^dense_200/MatMul/ReadVariableOp,^lstm_66/lstm_cell_66/BiasAdd/ReadVariableOp+^lstm_66/lstm_cell_66/MatMul/ReadVariableOp-^lstm_66/lstm_cell_66/MatMul_1/ReadVariableOp^lstm_66/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):���������: : : : : : : : : 2D
 dense_198/BiasAdd/ReadVariableOp dense_198/BiasAdd/ReadVariableOp2B
dense_198/MatMul/ReadVariableOpdense_198/MatMul/ReadVariableOp2D
 dense_199/BiasAdd/ReadVariableOp dense_199/BiasAdd/ReadVariableOp2B
dense_199/MatMul/ReadVariableOpdense_199/MatMul/ReadVariableOp2D
 dense_200/BiasAdd/ReadVariableOp dense_200/BiasAdd/ReadVariableOp2B
dense_200/MatMul/ReadVariableOpdense_200/MatMul/ReadVariableOp2Z
+lstm_66/lstm_cell_66/BiasAdd/ReadVariableOp+lstm_66/lstm_cell_66/BiasAdd/ReadVariableOp2X
*lstm_66/lstm_cell_66/MatMul/ReadVariableOp*lstm_66/lstm_cell_66/MatMul/ReadVariableOp2\
,lstm_66/lstm_cell_66/MatMul_1/ReadVariableOp,lstm_66/lstm_cell_66/MatMul_1/ReadVariableOp2
lstm_66/whilelstm_66/while:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
H__inference_dense_199_layer_call_and_return_conditional_losses_437463391

inputs0
matmul_readvariableop_resource:22-
biasadd_readvariableop_resource:2
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:22*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������2a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������2w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������2: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������2
 
_user_specified_nameinputs
�B
�

lstm_66_while_body_437463916,
(lstm_66_while_lstm_66_while_loop_counter2
.lstm_66_while_lstm_66_while_maximum_iterations
lstm_66_while_placeholder
lstm_66_while_placeholder_1
lstm_66_while_placeholder_2
lstm_66_while_placeholder_3+
'lstm_66_while_lstm_66_strided_slice_1_0g
clstm_66_while_tensorarrayv2read_tensorlistgetitem_lstm_66_tensorarrayunstack_tensorlistfromtensor_0N
;lstm_66_while_lstm_cell_66_matmul_readvariableop_resource_0:	�P
=lstm_66_while_lstm_cell_66_matmul_1_readvariableop_resource_0:	d�K
<lstm_66_while_lstm_cell_66_biasadd_readvariableop_resource_0:	�
lstm_66_while_identity
lstm_66_while_identity_1
lstm_66_while_identity_2
lstm_66_while_identity_3
lstm_66_while_identity_4
lstm_66_while_identity_5)
%lstm_66_while_lstm_66_strided_slice_1e
alstm_66_while_tensorarrayv2read_tensorlistgetitem_lstm_66_tensorarrayunstack_tensorlistfromtensorL
9lstm_66_while_lstm_cell_66_matmul_readvariableop_resource:	�N
;lstm_66_while_lstm_cell_66_matmul_1_readvariableop_resource:	d�I
:lstm_66_while_lstm_cell_66_biasadd_readvariableop_resource:	���1lstm_66/while/lstm_cell_66/BiasAdd/ReadVariableOp�0lstm_66/while/lstm_cell_66/MatMul/ReadVariableOp�2lstm_66/while/lstm_cell_66/MatMul_1/ReadVariableOp�
?lstm_66/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
1lstm_66/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_66_while_tensorarrayv2read_tensorlistgetitem_lstm_66_tensorarrayunstack_tensorlistfromtensor_0lstm_66_while_placeholderHlstm_66/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
0lstm_66/while/lstm_cell_66/MatMul/ReadVariableOpReadVariableOp;lstm_66_while_lstm_cell_66_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
!lstm_66/while/lstm_cell_66/MatMulMatMul8lstm_66/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_66/while/lstm_cell_66/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
2lstm_66/while/lstm_cell_66/MatMul_1/ReadVariableOpReadVariableOp=lstm_66_while_lstm_cell_66_matmul_1_readvariableop_resource_0*
_output_shapes
:	d�*
dtype0�
#lstm_66/while/lstm_cell_66/MatMul_1MatMullstm_66_while_placeholder_2:lstm_66/while/lstm_cell_66/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
lstm_66/while/lstm_cell_66/addAddV2+lstm_66/while/lstm_cell_66/MatMul:product:0-lstm_66/while/lstm_cell_66/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
1lstm_66/while/lstm_cell_66/BiasAdd/ReadVariableOpReadVariableOp<lstm_66_while_lstm_cell_66_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype0�
"lstm_66/while/lstm_cell_66/BiasAddBiasAdd"lstm_66/while/lstm_cell_66/add:z:09lstm_66/while/lstm_cell_66/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������l
*lstm_66/while/lstm_cell_66/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
 lstm_66/while/lstm_cell_66/splitSplit3lstm_66/while/lstm_cell_66/split/split_dim:output:0+lstm_66/while/lstm_cell_66/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������d:���������d:���������d:���������d*
	num_split�
"lstm_66/while/lstm_cell_66/SigmoidSigmoid)lstm_66/while/lstm_cell_66/split:output:0*
T0*'
_output_shapes
:���������d�
$lstm_66/while/lstm_cell_66/Sigmoid_1Sigmoid)lstm_66/while/lstm_cell_66/split:output:1*
T0*'
_output_shapes
:���������d�
lstm_66/while/lstm_cell_66/mulMul(lstm_66/while/lstm_cell_66/Sigmoid_1:y:0lstm_66_while_placeholder_3*
T0*'
_output_shapes
:���������d�
lstm_66/while/lstm_cell_66/ReluRelu)lstm_66/while/lstm_cell_66/split:output:2*
T0*'
_output_shapes
:���������d�
 lstm_66/while/lstm_cell_66/mul_1Mul&lstm_66/while/lstm_cell_66/Sigmoid:y:0-lstm_66/while/lstm_cell_66/Relu:activations:0*
T0*'
_output_shapes
:���������d�
 lstm_66/while/lstm_cell_66/add_1AddV2"lstm_66/while/lstm_cell_66/mul:z:0$lstm_66/while/lstm_cell_66/mul_1:z:0*
T0*'
_output_shapes
:���������d�
$lstm_66/while/lstm_cell_66/Sigmoid_2Sigmoid)lstm_66/while/lstm_cell_66/split:output:3*
T0*'
_output_shapes
:���������d�
!lstm_66/while/lstm_cell_66/Relu_1Relu$lstm_66/while/lstm_cell_66/add_1:z:0*
T0*'
_output_shapes
:���������d�
 lstm_66/while/lstm_cell_66/mul_2Mul(lstm_66/while/lstm_cell_66/Sigmoid_2:y:0/lstm_66/while/lstm_cell_66/Relu_1:activations:0*
T0*'
_output_shapes
:���������dz
8lstm_66/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : �
2lstm_66/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_66_while_placeholder_1Alstm_66/while/TensorArrayV2Write/TensorListSetItem/index:output:0$lstm_66/while/lstm_cell_66/mul_2:z:0*
_output_shapes
: *
element_dtype0:���U
lstm_66/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :t
lstm_66/while/addAddV2lstm_66_while_placeholderlstm_66/while/add/y:output:0*
T0*
_output_shapes
: W
lstm_66/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_66/while/add_1AddV2(lstm_66_while_lstm_66_while_loop_counterlstm_66/while/add_1/y:output:0*
T0*
_output_shapes
: q
lstm_66/while/IdentityIdentitylstm_66/while/add_1:z:0^lstm_66/while/NoOp*
T0*
_output_shapes
: �
lstm_66/while/Identity_1Identity.lstm_66_while_lstm_66_while_maximum_iterations^lstm_66/while/NoOp*
T0*
_output_shapes
: q
lstm_66/while/Identity_2Identitylstm_66/while/add:z:0^lstm_66/while/NoOp*
T0*
_output_shapes
: �
lstm_66/while/Identity_3IdentityBlstm_66/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_66/while/NoOp*
T0*
_output_shapes
: �
lstm_66/while/Identity_4Identity$lstm_66/while/lstm_cell_66/mul_2:z:0^lstm_66/while/NoOp*
T0*'
_output_shapes
:���������d�
lstm_66/while/Identity_5Identity$lstm_66/while/lstm_cell_66/add_1:z:0^lstm_66/while/NoOp*
T0*'
_output_shapes
:���������d�
lstm_66/while/NoOpNoOp2^lstm_66/while/lstm_cell_66/BiasAdd/ReadVariableOp1^lstm_66/while/lstm_cell_66/MatMul/ReadVariableOp3^lstm_66/while/lstm_cell_66/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "9
lstm_66_while_identitylstm_66/while/Identity:output:0"=
lstm_66_while_identity_1!lstm_66/while/Identity_1:output:0"=
lstm_66_while_identity_2!lstm_66/while/Identity_2:output:0"=
lstm_66_while_identity_3!lstm_66/while/Identity_3:output:0"=
lstm_66_while_identity_4!lstm_66/while/Identity_4:output:0"=
lstm_66_while_identity_5!lstm_66/while/Identity_5:output:0"P
%lstm_66_while_lstm_66_strided_slice_1'lstm_66_while_lstm_66_strided_slice_1_0"z
:lstm_66_while_lstm_cell_66_biasadd_readvariableop_resource<lstm_66_while_lstm_cell_66_biasadd_readvariableop_resource_0"|
;lstm_66_while_lstm_cell_66_matmul_1_readvariableop_resource=lstm_66_while_lstm_cell_66_matmul_1_readvariableop_resource_0"x
9lstm_66_while_lstm_cell_66_matmul_readvariableop_resource;lstm_66_while_lstm_cell_66_matmul_readvariableop_resource_0"�
alstm_66_while_tensorarrayv2read_tensorlistgetitem_lstm_66_tensorarrayunstack_tensorlistfromtensorclstm_66_while_tensorarrayv2read_tensorlistgetitem_lstm_66_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������d:���������d: : : : : 2f
1lstm_66/while/lstm_cell_66/BiasAdd/ReadVariableOp1lstm_66/while/lstm_cell_66/BiasAdd/ReadVariableOp2d
0lstm_66/while/lstm_cell_66/MatMul/ReadVariableOp0lstm_66/while/lstm_cell_66/MatMul/ReadVariableOp2h
2lstm_66/while/lstm_cell_66/MatMul_1/ReadVariableOp2lstm_66/while/lstm_cell_66/MatMul_1/ReadVariableOp: 
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
:���������d:-)
'
_output_shapes
:���������d:

_output_shapes
: :

_output_shapes
: 
�
�
while_cond_437463269
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_17
3while_while_cond_437463269___redundant_placeholder07
3while_while_cond_437463269___redundant_placeholder17
3while_while_cond_437463269___redundant_placeholder27
3while_while_cond_437463269___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
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
@: : : : :���������d:���������d: ::::: 
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
:���������d:-)
'
_output_shapes
:���������d:

_output_shapes
: :

_output_shapes
:
�
�
-__inference_dense_200_layer_call_fn_437464859

inputs
unknown:2
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_dense_200_layer_call_and_return_conditional_losses_437463407o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������2: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������2
 
_user_specified_nameinputs"�	L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
K
lstm_66_input:
serving_default_lstm_66_input:0���������=
	dense_2000
StatefulPartitionedCall:0���������tensorflow/serving/predict:��
�
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
	variables
trainable_variables
regularization_losses
	keras_api
	__call__
*
&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_sequential
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_random_generator
cell

state_spec"
_tf_keras_rnn_layer
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
�
	variables
 trainable_variables
!regularization_losses
"	keras_api
#__call__
*$&call_and_return_all_conditional_losses

%kernel
&bias"
_tf_keras_layer
�
'	variables
(trainable_variables
)regularization_losses
*	keras_api
+__call__
*,&call_and_return_all_conditional_losses

-kernel
.bias"
_tf_keras_layer
_
/0
01
12
3
4
%5
&6
-7
.8"
trackable_list_wrapper
_
/0
01
12
3
4
%5
&6
-7
.8"
trackable_list_wrapper
 "
trackable_list_wrapper
�
2non_trainable_variables

3layers
4metrics
5layer_regularization_losses
6layer_metrics
	variables
trainable_variables
regularization_losses
	__call__
_default_save_signature
*
&call_and_return_all_conditional_losses
&
"call_and_return_conditional_losses"
_generic_user_object
�
7trace_0
8trace_1
9trace_2
:trace_32�
1__inference_sequential_66_layer_call_fn_437463435
1__inference_sequential_66_layer_call_fn_437463833
1__inference_sequential_66_layer_call_fn_437463856
1__inference_sequential_66_layer_call_fn_437463727�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z7trace_0z8trace_1z9trace_2z:trace_3
�
;trace_0
<trace_1
=trace_2
>trace_32�
L__inference_sequential_66_layer_call_and_return_conditional_losses_437464021
L__inference_sequential_66_layer_call_and_return_conditional_losses_437464186
L__inference_sequential_66_layer_call_and_return_conditional_losses_437463753
L__inference_sequential_66_layer_call_and_return_conditional_losses_437463779�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z;trace_0z<trace_1z=trace_2z>trace_3
�B�
$__inference__wrapped_model_437462850lstm_66_input"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
?iter

@beta_1

Abeta_2
	Bdecay
Clearning_ratem�m�%m�&m�-m�.m�/m�0m�1m�v�v�%v�&v�-v�.v�/v�0v�1v�"
	optimizer
,
Dserving_default"
signature_map
5
/0
01
12"
trackable_list_wrapper
5
/0
01
12"
trackable_list_wrapper
 "
trackable_list_wrapper
�

Estates
Fnon_trainable_variables

Glayers
Hmetrics
Ilayer_regularization_losses
Jlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
Ktrace_0
Ltrace_1
Mtrace_2
Ntrace_32�
+__inference_lstm_66_layer_call_fn_437464197
+__inference_lstm_66_layer_call_fn_437464208
+__inference_lstm_66_layer_call_fn_437464219
+__inference_lstm_66_layer_call_fn_437464230�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zKtrace_0zLtrace_1zMtrace_2zNtrace_3
�
Otrace_0
Ptrace_1
Qtrace_2
Rtrace_32�
F__inference_lstm_66_layer_call_and_return_conditional_losses_437464375
F__inference_lstm_66_layer_call_and_return_conditional_losses_437464520
F__inference_lstm_66_layer_call_and_return_conditional_losses_437464665
F__inference_lstm_66_layer_call_and_return_conditional_losses_437464810�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zOtrace_0zPtrace_1zQtrace_2zRtrace_3
"
_generic_user_object
�
S	variables
Ttrainable_variables
Uregularization_losses
V	keras_api
W__call__
*X&call_and_return_all_conditional_losses
Y_random_generator
Z
state_size

/kernel
0recurrent_kernel
1bias"
_tf_keras_layer
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
[non_trainable_variables

\layers
]metrics
^layer_regularization_losses
_layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
`trace_02�
-__inference_dense_198_layer_call_fn_437464819�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z`trace_0
�
atrace_02�
H__inference_dense_198_layer_call_and_return_conditional_losses_437464830�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zatrace_0
": d22dense_198/kernel
:22dense_198/bias
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
�
bnon_trainable_variables

clayers
dmetrics
elayer_regularization_losses
flayer_metrics
	variables
 trainable_variables
!regularization_losses
#__call__
*$&call_and_return_all_conditional_losses
&$"call_and_return_conditional_losses"
_generic_user_object
�
gtrace_02�
-__inference_dense_199_layer_call_fn_437464839�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zgtrace_0
�
htrace_02�
H__inference_dense_199_layer_call_and_return_conditional_losses_437464850�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zhtrace_0
": 222dense_199/kernel
:22dense_199/bias
.
-0
.1"
trackable_list_wrapper
.
-0
.1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
inon_trainable_variables

jlayers
kmetrics
llayer_regularization_losses
mlayer_metrics
'	variables
(trainable_variables
)regularization_losses
+__call__
*,&call_and_return_all_conditional_losses
&,"call_and_return_conditional_losses"
_generic_user_object
�
ntrace_02�
-__inference_dense_200_layer_call_fn_437464859�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zntrace_0
�
otrace_02�
H__inference_dense_200_layer_call_and_return_conditional_losses_437464869�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zotrace_0
": 22dense_200/kernel
:2dense_200/bias
.:,	�2lstm_66/lstm_cell_66/kernel
8:6	d�2%lstm_66/lstm_cell_66/recurrent_kernel
(:&�2lstm_66/lstm_cell_66/bias
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
.
p0
q1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
1__inference_sequential_66_layer_call_fn_437463435lstm_66_input"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
1__inference_sequential_66_layer_call_fn_437463833inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
1__inference_sequential_66_layer_call_fn_437463856inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
1__inference_sequential_66_layer_call_fn_437463727lstm_66_input"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
L__inference_sequential_66_layer_call_and_return_conditional_losses_437464021inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
L__inference_sequential_66_layer_call_and_return_conditional_losses_437464186inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
L__inference_sequential_66_layer_call_and_return_conditional_losses_437463753lstm_66_input"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
L__inference_sequential_66_layer_call_and_return_conditional_losses_437463779lstm_66_input"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
�B�
'__inference_signature_wrapper_437463810lstm_66_input"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
+__inference_lstm_66_layer_call_fn_437464197inputs/0"�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
+__inference_lstm_66_layer_call_fn_437464208inputs/0"�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
+__inference_lstm_66_layer_call_fn_437464219inputs"�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
+__inference_lstm_66_layer_call_fn_437464230inputs"�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_lstm_66_layer_call_and_return_conditional_losses_437464375inputs/0"�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_lstm_66_layer_call_and_return_conditional_losses_437464520inputs/0"�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_lstm_66_layer_call_and_return_conditional_losses_437464665inputs"�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_lstm_66_layer_call_and_return_conditional_losses_437464810inputs"�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
5
/0
01
12"
trackable_list_wrapper
5
/0
01
12"
trackable_list_wrapper
 "
trackable_list_wrapper
�
rnon_trainable_variables

slayers
tmetrics
ulayer_regularization_losses
vlayer_metrics
S	variables
Ttrainable_variables
Uregularization_losses
W__call__
*X&call_and_return_all_conditional_losses
&X"call_and_return_conditional_losses"
_generic_user_object
�
wtrace_0
xtrace_12�
0__inference_lstm_cell_66_layer_call_fn_437464886
0__inference_lstm_cell_66_layer_call_fn_437464903�
���
FullArgSpec3
args+�(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zwtrace_0zxtrace_1
�
ytrace_0
ztrace_12�
K__inference_lstm_cell_66_layer_call_and_return_conditional_losses_437464935
K__inference_lstm_cell_66_layer_call_and_return_conditional_losses_437464967�
���
FullArgSpec3
args+�(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zytrace_0zztrace_1
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
�B�
-__inference_dense_198_layer_call_fn_437464819inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
H__inference_dense_198_layer_call_and_return_conditional_losses_437464830inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
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
�B�
-__inference_dense_199_layer_call_fn_437464839inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
H__inference_dense_199_layer_call_and_return_conditional_losses_437464850inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
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
�B�
-__inference_dense_200_layer_call_fn_437464859inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
H__inference_dense_200_layer_call_and_return_conditional_losses_437464869inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
N
{	variables
|	keras_api
	}total
	~count"
_tf_keras_metric
b
	variables
�	keras_api

�total

�count
�
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
�B�
0__inference_lstm_cell_66_layer_call_fn_437464886inputsstates/0states/1"�
���
FullArgSpec3
args+�(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
0__inference_lstm_cell_66_layer_call_fn_437464903inputsstates/0states/1"�
���
FullArgSpec3
args+�(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
K__inference_lstm_cell_66_layer_call_and_return_conditional_losses_437464935inputsstates/0states/1"�
���
FullArgSpec3
args+�(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
K__inference_lstm_cell_66_layer_call_and_return_conditional_losses_437464967inputsstates/0states/1"�
���
FullArgSpec3
args+�(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
.
}0
~1"
trackable_list_wrapper
-
{	variables"
_generic_user_object
:  (2total
:  (2count
0
�0
�1"
trackable_list_wrapper
-
	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
':%d22Adam/dense_198/kernel/m
!:22Adam/dense_198/bias/m
':%222Adam/dense_199/kernel/m
!:22Adam/dense_199/bias/m
':%22Adam/dense_200/kernel/m
!:2Adam/dense_200/bias/m
3:1	�2"Adam/lstm_66/lstm_cell_66/kernel/m
=:;	d�2,Adam/lstm_66/lstm_cell_66/recurrent_kernel/m
-:+�2 Adam/lstm_66/lstm_cell_66/bias/m
':%d22Adam/dense_198/kernel/v
!:22Adam/dense_198/bias/v
':%222Adam/dense_199/kernel/v
!:22Adam/dense_199/bias/v
':%22Adam/dense_200/kernel/v
!:2Adam/dense_200/bias/v
3:1	�2"Adam/lstm_66/lstm_cell_66/kernel/v
=:;	d�2,Adam/lstm_66/lstm_cell_66/recurrent_kernel/v
-:+�2 Adam/lstm_66/lstm_cell_66/bias/v�
$__inference__wrapped_model_437462850~	/01%&-.:�7
0�-
+�(
lstm_66_input���������
� "5�2
0
	dense_200#� 
	dense_200����������
H__inference_dense_198_layer_call_and_return_conditional_losses_437464830\/�,
%�"
 �
inputs���������d
� "%�"
�
0���������2
� �
-__inference_dense_198_layer_call_fn_437464819O/�,
%�"
 �
inputs���������d
� "����������2�
H__inference_dense_199_layer_call_and_return_conditional_losses_437464850\%&/�,
%�"
 �
inputs���������2
� "%�"
�
0���������2
� �
-__inference_dense_199_layer_call_fn_437464839O%&/�,
%�"
 �
inputs���������2
� "����������2�
H__inference_dense_200_layer_call_and_return_conditional_losses_437464869\-./�,
%�"
 �
inputs���������2
� "%�"
�
0���������
� �
-__inference_dense_200_layer_call_fn_437464859O-./�,
%�"
 �
inputs���������2
� "�����������
F__inference_lstm_66_layer_call_and_return_conditional_losses_437464375}/01O�L
E�B
4�1
/�,
inputs/0������������������

 
p 

 
� "%�"
�
0���������d
� �
F__inference_lstm_66_layer_call_and_return_conditional_losses_437464520}/01O�L
E�B
4�1
/�,
inputs/0������������������

 
p

 
� "%�"
�
0���������d
� �
F__inference_lstm_66_layer_call_and_return_conditional_losses_437464665m/01?�<
5�2
$�!
inputs���������

 
p 

 
� "%�"
�
0���������d
� �
F__inference_lstm_66_layer_call_and_return_conditional_losses_437464810m/01?�<
5�2
$�!
inputs���������

 
p

 
� "%�"
�
0���������d
� �
+__inference_lstm_66_layer_call_fn_437464197p/01O�L
E�B
4�1
/�,
inputs/0������������������

 
p 

 
� "����������d�
+__inference_lstm_66_layer_call_fn_437464208p/01O�L
E�B
4�1
/�,
inputs/0������������������

 
p

 
� "����������d�
+__inference_lstm_66_layer_call_fn_437464219`/01?�<
5�2
$�!
inputs���������

 
p 

 
� "����������d�
+__inference_lstm_66_layer_call_fn_437464230`/01?�<
5�2
$�!
inputs���������

 
p

 
� "����������d�
K__inference_lstm_cell_66_layer_call_and_return_conditional_losses_437464935�/01��}
v�s
 �
inputs���������
K�H
"�
states/0���������d
"�
states/1���������d
p 
� "s�p
i�f
�
0/0���������d
E�B
�
0/1/0���������d
�
0/1/1���������d
� �
K__inference_lstm_cell_66_layer_call_and_return_conditional_losses_437464967�/01��}
v�s
 �
inputs���������
K�H
"�
states/0���������d
"�
states/1���������d
p
� "s�p
i�f
�
0/0���������d
E�B
�
0/1/0���������d
�
0/1/1���������d
� �
0__inference_lstm_cell_66_layer_call_fn_437464886�/01��}
v�s
 �
inputs���������
K�H
"�
states/0���������d
"�
states/1���������d
p 
� "c�`
�
0���������d
A�>
�
1/0���������d
�
1/1���������d�
0__inference_lstm_cell_66_layer_call_fn_437464903�/01��}
v�s
 �
inputs���������
K�H
"�
states/0���������d
"�
states/1���������d
p
� "c�`
�
0���������d
A�>
�
1/0���������d
�
1/1���������d�
L__inference_sequential_66_layer_call_and_return_conditional_losses_437463753v	/01%&-.B�?
8�5
+�(
lstm_66_input���������
p 

 
� "%�"
�
0���������
� �
L__inference_sequential_66_layer_call_and_return_conditional_losses_437463779v	/01%&-.B�?
8�5
+�(
lstm_66_input���������
p

 
� "%�"
�
0���������
� �
L__inference_sequential_66_layer_call_and_return_conditional_losses_437464021o	/01%&-.;�8
1�.
$�!
inputs���������
p 

 
� "%�"
�
0���������
� �
L__inference_sequential_66_layer_call_and_return_conditional_losses_437464186o	/01%&-.;�8
1�.
$�!
inputs���������
p

 
� "%�"
�
0���������
� �
1__inference_sequential_66_layer_call_fn_437463435i	/01%&-.B�?
8�5
+�(
lstm_66_input���������
p 

 
� "�����������
1__inference_sequential_66_layer_call_fn_437463727i	/01%&-.B�?
8�5
+�(
lstm_66_input���������
p

 
� "�����������
1__inference_sequential_66_layer_call_fn_437463833b	/01%&-.;�8
1�.
$�!
inputs���������
p 

 
� "�����������
1__inference_sequential_66_layer_call_fn_437463856b	/01%&-.;�8
1�.
$�!
inputs���������
p

 
� "�����������
'__inference_signature_wrapper_437463810�	/01%&-.K�H
� 
A�>
<
lstm_66_input+�(
lstm_66_input���������"5�2
0
	dense_200#� 
	dense_200���������