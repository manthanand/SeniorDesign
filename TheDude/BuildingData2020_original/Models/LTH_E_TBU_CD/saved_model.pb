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
 Adam/lstm_40/lstm_cell_40/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*1
shared_name" Adam/lstm_40/lstm_cell_40/bias/v
�
4Adam/lstm_40/lstm_cell_40/bias/v/Read/ReadVariableOpReadVariableOp Adam/lstm_40/lstm_cell_40/bias/v*
_output_shapes	
:�*
dtype0
�
,Adam/lstm_40/lstm_cell_40/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	d�*=
shared_name.,Adam/lstm_40/lstm_cell_40/recurrent_kernel/v
�
@Adam/lstm_40/lstm_cell_40/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp,Adam/lstm_40/lstm_cell_40/recurrent_kernel/v*
_output_shapes
:	d�*
dtype0
�
"Adam/lstm_40/lstm_cell_40/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*3
shared_name$"Adam/lstm_40/lstm_cell_40/kernel/v
�
6Adam/lstm_40/lstm_cell_40/kernel/v/Read/ReadVariableOpReadVariableOp"Adam/lstm_40/lstm_cell_40/kernel/v*
_output_shapes
:	�*
dtype0
�
Adam/dense_122/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_122/bias/v
{
)Adam/dense_122/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_122/bias/v*
_output_shapes
:*
dtype0
�
Adam/dense_122/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*(
shared_nameAdam/dense_122/kernel/v
�
+Adam/dense_122/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_122/kernel/v*
_output_shapes

:2*
dtype0
�
Adam/dense_121/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*&
shared_nameAdam/dense_121/bias/v
{
)Adam/dense_121/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_121/bias/v*
_output_shapes
:2*
dtype0
�
Adam/dense_121/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:22*(
shared_nameAdam/dense_121/kernel/v
�
+Adam/dense_121/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_121/kernel/v*
_output_shapes

:22*
dtype0
�
Adam/dense_120/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*&
shared_nameAdam/dense_120/bias/v
{
)Adam/dense_120/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_120/bias/v*
_output_shapes
:2*
dtype0
�
Adam/dense_120/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d2*(
shared_nameAdam/dense_120/kernel/v
�
+Adam/dense_120/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_120/kernel/v*
_output_shapes

:d2*
dtype0
�
 Adam/lstm_40/lstm_cell_40/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*1
shared_name" Adam/lstm_40/lstm_cell_40/bias/m
�
4Adam/lstm_40/lstm_cell_40/bias/m/Read/ReadVariableOpReadVariableOp Adam/lstm_40/lstm_cell_40/bias/m*
_output_shapes	
:�*
dtype0
�
,Adam/lstm_40/lstm_cell_40/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	d�*=
shared_name.,Adam/lstm_40/lstm_cell_40/recurrent_kernel/m
�
@Adam/lstm_40/lstm_cell_40/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp,Adam/lstm_40/lstm_cell_40/recurrent_kernel/m*
_output_shapes
:	d�*
dtype0
�
"Adam/lstm_40/lstm_cell_40/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*3
shared_name$"Adam/lstm_40/lstm_cell_40/kernel/m
�
6Adam/lstm_40/lstm_cell_40/kernel/m/Read/ReadVariableOpReadVariableOp"Adam/lstm_40/lstm_cell_40/kernel/m*
_output_shapes
:	�*
dtype0
�
Adam/dense_122/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_122/bias/m
{
)Adam/dense_122/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_122/bias/m*
_output_shapes
:*
dtype0
�
Adam/dense_122/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*(
shared_nameAdam/dense_122/kernel/m
�
+Adam/dense_122/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_122/kernel/m*
_output_shapes

:2*
dtype0
�
Adam/dense_121/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*&
shared_nameAdam/dense_121/bias/m
{
)Adam/dense_121/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_121/bias/m*
_output_shapes
:2*
dtype0
�
Adam/dense_121/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:22*(
shared_nameAdam/dense_121/kernel/m
�
+Adam/dense_121/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_121/kernel/m*
_output_shapes

:22*
dtype0
�
Adam/dense_120/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*&
shared_nameAdam/dense_120/bias/m
{
)Adam/dense_120/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_120/bias/m*
_output_shapes
:2*
dtype0
�
Adam/dense_120/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d2*(
shared_nameAdam/dense_120/kernel/m
�
+Adam/dense_120/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_120/kernel/m*
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
lstm_40/lstm_cell_40/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�**
shared_namelstm_40/lstm_cell_40/bias
�
-lstm_40/lstm_cell_40/bias/Read/ReadVariableOpReadVariableOplstm_40/lstm_cell_40/bias*
_output_shapes	
:�*
dtype0
�
%lstm_40/lstm_cell_40/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	d�*6
shared_name'%lstm_40/lstm_cell_40/recurrent_kernel
�
9lstm_40/lstm_cell_40/recurrent_kernel/Read/ReadVariableOpReadVariableOp%lstm_40/lstm_cell_40/recurrent_kernel*
_output_shapes
:	d�*
dtype0
�
lstm_40/lstm_cell_40/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*,
shared_namelstm_40/lstm_cell_40/kernel
�
/lstm_40/lstm_cell_40/kernel/Read/ReadVariableOpReadVariableOplstm_40/lstm_cell_40/kernel*
_output_shapes
:	�*
dtype0
t
dense_122/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_122/bias
m
"dense_122/bias/Read/ReadVariableOpReadVariableOpdense_122/bias*
_output_shapes
:*
dtype0
|
dense_122/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*!
shared_namedense_122/kernel
u
$dense_122/kernel/Read/ReadVariableOpReadVariableOpdense_122/kernel*
_output_shapes

:2*
dtype0
t
dense_121/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*
shared_namedense_121/bias
m
"dense_121/bias/Read/ReadVariableOpReadVariableOpdense_121/bias*
_output_shapes
:2*
dtype0
|
dense_121/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:22*!
shared_namedense_121/kernel
u
$dense_121/kernel/Read/ReadVariableOpReadVariableOpdense_121/kernel*
_output_shapes

:22*
dtype0
t
dense_120/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*
shared_namedense_120/bias
m
"dense_120/bias/Read/ReadVariableOpReadVariableOpdense_120/bias*
_output_shapes
:2*
dtype0
|
dense_120/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d2*!
shared_namedense_120/kernel
u
$dense_120/kernel/Read/ReadVariableOpReadVariableOpdense_120/kernel*
_output_shapes

:d2*
dtype0
�
serving_default_lstm_40_inputPlaceholder*+
_output_shapes
:���������*
dtype0* 
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_lstm_40_inputlstm_40/lstm_cell_40/kernel%lstm_40/lstm_cell_40/recurrent_kernellstm_40/lstm_cell_40/biasdense_120/kerneldense_120/biasdense_121/kerneldense_121/biasdense_122/kerneldense_122/bias*
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
GPU 2J 8� *.
f)R'
%__inference_signature_wrapper_1378285

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
VARIABLE_VALUEdense_120/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_120/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEdense_121/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_121/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEdense_122/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_122/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUElstm_40/lstm_cell_40/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUE%lstm_40/lstm_cell_40/recurrent_kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUElstm_40/lstm_cell_40/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEAdam/dense_120/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_120/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�}
VARIABLE_VALUEAdam/dense_121/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_121/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�}
VARIABLE_VALUEAdam/dense_122/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_122/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE"Adam/lstm_40/lstm_cell_40/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE,Adam/lstm_40/lstm_cell_40/recurrent_kernel/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE Adam/lstm_40/lstm_cell_40/bias/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�}
VARIABLE_VALUEAdam/dense_120/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_120/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�}
VARIABLE_VALUEAdam/dense_121/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_121/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�}
VARIABLE_VALUEAdam/dense_122/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_122/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE"Adam/lstm_40/lstm_cell_40/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE,Adam/lstm_40/lstm_cell_40/recurrent_kernel/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE Adam/lstm_40/lstm_cell_40/bias/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$dense_120/kernel/Read/ReadVariableOp"dense_120/bias/Read/ReadVariableOp$dense_121/kernel/Read/ReadVariableOp"dense_121/bias/Read/ReadVariableOp$dense_122/kernel/Read/ReadVariableOp"dense_122/bias/Read/ReadVariableOp/lstm_40/lstm_cell_40/kernel/Read/ReadVariableOp9lstm_40/lstm_cell_40/recurrent_kernel/Read/ReadVariableOp-lstm_40/lstm_cell_40/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp+Adam/dense_120/kernel/m/Read/ReadVariableOp)Adam/dense_120/bias/m/Read/ReadVariableOp+Adam/dense_121/kernel/m/Read/ReadVariableOp)Adam/dense_121/bias/m/Read/ReadVariableOp+Adam/dense_122/kernel/m/Read/ReadVariableOp)Adam/dense_122/bias/m/Read/ReadVariableOp6Adam/lstm_40/lstm_cell_40/kernel/m/Read/ReadVariableOp@Adam/lstm_40/lstm_cell_40/recurrent_kernel/m/Read/ReadVariableOp4Adam/lstm_40/lstm_cell_40/bias/m/Read/ReadVariableOp+Adam/dense_120/kernel/v/Read/ReadVariableOp)Adam/dense_120/bias/v/Read/ReadVariableOp+Adam/dense_121/kernel/v/Read/ReadVariableOp)Adam/dense_121/bias/v/Read/ReadVariableOp+Adam/dense_122/kernel/v/Read/ReadVariableOp)Adam/dense_122/bias/v/Read/ReadVariableOp6Adam/lstm_40/lstm_cell_40/kernel/v/Read/ReadVariableOp@Adam/lstm_40/lstm_cell_40/recurrent_kernel/v/Read/ReadVariableOp4Adam/lstm_40/lstm_cell_40/bias/v/Read/ReadVariableOpConst*1
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
GPU 2J 8� *)
f$R"
 __inference__traced_save_1379573
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_120/kerneldense_120/biasdense_121/kerneldense_121/biasdense_122/kerneldense_122/biaslstm_40/lstm_cell_40/kernel%lstm_40/lstm_cell_40/recurrent_kernellstm_40/lstm_cell_40/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotal_1count_1totalcountAdam/dense_120/kernel/mAdam/dense_120/bias/mAdam/dense_121/kernel/mAdam/dense_121/bias/mAdam/dense_122/kernel/mAdam/dense_122/bias/m"Adam/lstm_40/lstm_cell_40/kernel/m,Adam/lstm_40/lstm_cell_40/recurrent_kernel/m Adam/lstm_40/lstm_cell_40/bias/mAdam/dense_120/kernel/vAdam/dense_120/bias/vAdam/dense_121/kernel/vAdam/dense_121/bias/vAdam/dense_122/kernel/vAdam/dense_122/bias/v"Adam/lstm_40/lstm_cell_40/kernel/v,Adam/lstm_40/lstm_cell_40/recurrent_kernel/v Adam/lstm_40/lstm_cell_40/bias/v*0
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
GPU 2J 8� *,
f'R%
#__inference__traced_restore_1379691��
�
�
while_cond_1377599
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1377599___redundant_placeholder05
1while_while_cond_1377599___redundant_placeholder15
1while_while_cond_1377599___redundant_placeholder25
1while_while_cond_1377599___redundant_placeholder3
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
�
�
I__inference_lstm_cell_40_layer_call_and_return_conditional_losses_1377540

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
�
�
.__inference_lstm_cell_40_layer_call_fn_1379378

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
GPU 2J 8� *R
fMRK
I__inference_lstm_cell_40_layer_call_and_return_conditional_losses_1377540o
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
F__inference_dense_122_layer_call_and_return_conditional_losses_1379344

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
�
�
)__inference_lstm_40_layer_call_fn_1378694

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
GPU 2J 8� *M
fHRF
D__inference_lstm_40_layer_call_and_return_conditional_losses_1377830o
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
�	
�
F__inference_dense_122_layer_call_and_return_conditional_losses_1377882

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
�
�
)__inference_lstm_40_layer_call_fn_1378683
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
GPU 2J 8� *M
fHRF
D__inference_lstm_40_layer_call_and_return_conditional_losses_1377670o
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
�K
�
D__inference_lstm_40_layer_call_and_return_conditional_losses_1379285

inputs>
+lstm_cell_40_matmul_readvariableop_resource:	�@
-lstm_cell_40_matmul_1_readvariableop_resource:	d�;
,lstm_cell_40_biasadd_readvariableop_resource:	�
identity��#lstm_cell_40/BiasAdd/ReadVariableOp�"lstm_cell_40/MatMul/ReadVariableOp�$lstm_cell_40/MatMul_1/ReadVariableOp�while;
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
"lstm_cell_40/MatMul/ReadVariableOpReadVariableOp+lstm_cell_40_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
lstm_cell_40/MatMulMatMulstrided_slice_2:output:0*lstm_cell_40/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
$lstm_cell_40/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_40_matmul_1_readvariableop_resource*
_output_shapes
:	d�*
dtype0�
lstm_cell_40/MatMul_1MatMulzeros:output:0,lstm_cell_40/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
lstm_cell_40/addAddV2lstm_cell_40/MatMul:product:0lstm_cell_40/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
#lstm_cell_40/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_40_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
lstm_cell_40/BiasAddBiasAddlstm_cell_40/add:z:0+lstm_cell_40/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������^
lstm_cell_40/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_cell_40/splitSplit%lstm_cell_40/split/split_dim:output:0lstm_cell_40/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������d:���������d:���������d:���������d*
	num_splitn
lstm_cell_40/SigmoidSigmoidlstm_cell_40/split:output:0*
T0*'
_output_shapes
:���������dp
lstm_cell_40/Sigmoid_1Sigmoidlstm_cell_40/split:output:1*
T0*'
_output_shapes
:���������dw
lstm_cell_40/mulMullstm_cell_40/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������dh
lstm_cell_40/ReluRelulstm_cell_40/split:output:2*
T0*'
_output_shapes
:���������d�
lstm_cell_40/mul_1Mullstm_cell_40/Sigmoid:y:0lstm_cell_40/Relu:activations:0*
T0*'
_output_shapes
:���������d{
lstm_cell_40/add_1AddV2lstm_cell_40/mul:z:0lstm_cell_40/mul_1:z:0*
T0*'
_output_shapes
:���������dp
lstm_cell_40/Sigmoid_2Sigmoidlstm_cell_40/split:output:3*
T0*'
_output_shapes
:���������de
lstm_cell_40/Relu_1Relulstm_cell_40/add_1:z:0*
T0*'
_output_shapes
:���������d�
lstm_cell_40/mul_2Mullstm_cell_40/Sigmoid_2:y:0!lstm_cell_40/Relu_1:activations:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_40_matmul_readvariableop_resource-lstm_cell_40_matmul_1_readvariableop_resource,lstm_cell_40_biasadd_readvariableop_resource*
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
bodyR
while_body_1379200*
condR
while_cond_1379199*K
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
NoOpNoOp$^lstm_cell_40/BiasAdd/ReadVariableOp#^lstm_cell_40/MatMul/ReadVariableOp%^lstm_cell_40/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������: : : 2J
#lstm_cell_40/BiasAdd/ReadVariableOp#lstm_cell_40/BiasAdd/ReadVariableOp2H
"lstm_cell_40/MatMul/ReadVariableOp"lstm_cell_40/MatMul/ReadVariableOp2L
$lstm_cell_40/MatMul_1/ReadVariableOp$lstm_cell_40/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
I__inference_lstm_cell_40_layer_call_and_return_conditional_losses_1377392

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
�$
�
while_body_1377407
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0/
while_lstm_cell_40_1377431_0:	�/
while_lstm_cell_40_1377433_0:	d�+
while_lstm_cell_40_1377435_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor-
while_lstm_cell_40_1377431:	�-
while_lstm_cell_40_1377433:	d�)
while_lstm_cell_40_1377435:	���*while/lstm_cell_40/StatefulPartitionedCall�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
*while/lstm_cell_40/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_40_1377431_0while_lstm_cell_40_1377433_0while_lstm_cell_40_1377435_0*
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
GPU 2J 8� *R
fMRK
I__inference_lstm_cell_40_layer_call_and_return_conditional_losses_1377392r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : �
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:03while/lstm_cell_40/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity3while/lstm_cell_40/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:���������d�
while/Identity_5Identity3while/lstm_cell_40/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:���������dy

while/NoOpNoOp+^while/lstm_cell_40/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0":
while_lstm_cell_40_1377431while_lstm_cell_40_1377431_0":
while_lstm_cell_40_1377433while_lstm_cell_40_1377433_0":
while_lstm_cell_40_1377435while_lstm_cell_40_1377435_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������d:���������d: : : : : 2X
*while/lstm_cell_40/StatefulPartitionedCall*while/lstm_cell_40/StatefulPartitionedCall: 
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
J__inference_sequential_40_layer_call_and_return_conditional_losses_1378661

inputsF
3lstm_40_lstm_cell_40_matmul_readvariableop_resource:	�H
5lstm_40_lstm_cell_40_matmul_1_readvariableop_resource:	d�C
4lstm_40_lstm_cell_40_biasadd_readvariableop_resource:	�:
(dense_120_matmul_readvariableop_resource:d27
)dense_120_biasadd_readvariableop_resource:2:
(dense_121_matmul_readvariableop_resource:227
)dense_121_biasadd_readvariableop_resource:2:
(dense_122_matmul_readvariableop_resource:27
)dense_122_biasadd_readvariableop_resource:
identity�� dense_120/BiasAdd/ReadVariableOp�dense_120/MatMul/ReadVariableOp� dense_121/BiasAdd/ReadVariableOp�dense_121/MatMul/ReadVariableOp� dense_122/BiasAdd/ReadVariableOp�dense_122/MatMul/ReadVariableOp�+lstm_40/lstm_cell_40/BiasAdd/ReadVariableOp�*lstm_40/lstm_cell_40/MatMul/ReadVariableOp�,lstm_40/lstm_cell_40/MatMul_1/ReadVariableOp�lstm_40/whileC
lstm_40/ShapeShapeinputs*
T0*
_output_shapes
:e
lstm_40/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
lstm_40/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
lstm_40/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
lstm_40/strided_sliceStridedSlicelstm_40/Shape:output:0$lstm_40/strided_slice/stack:output:0&lstm_40/strided_slice/stack_1:output:0&lstm_40/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
lstm_40/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d�
lstm_40/zeros/packedPacklstm_40/strided_slice:output:0lstm_40/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:X
lstm_40/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
lstm_40/zerosFilllstm_40/zeros/packed:output:0lstm_40/zeros/Const:output:0*
T0*'
_output_shapes
:���������dZ
lstm_40/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d�
lstm_40/zeros_1/packedPacklstm_40/strided_slice:output:0!lstm_40/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:Z
lstm_40/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
lstm_40/zeros_1Filllstm_40/zeros_1/packed:output:0lstm_40/zeros_1/Const:output:0*
T0*'
_output_shapes
:���������dk
lstm_40/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          }
lstm_40/transpose	Transposeinputslstm_40/transpose/perm:output:0*
T0*+
_output_shapes
:���������T
lstm_40/Shape_1Shapelstm_40/transpose:y:0*
T0*
_output_shapes
:g
lstm_40/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: i
lstm_40/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
lstm_40/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
lstm_40/strided_slice_1StridedSlicelstm_40/Shape_1:output:0&lstm_40/strided_slice_1/stack:output:0(lstm_40/strided_slice_1/stack_1:output:0(lstm_40/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskn
#lstm_40/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
lstm_40/TensorArrayV2TensorListReserve,lstm_40/TensorArrayV2/element_shape:output:0 lstm_40/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
=lstm_40/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
/lstm_40/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_40/transpose:y:0Flstm_40/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���g
lstm_40/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: i
lstm_40/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
lstm_40/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
lstm_40/strided_slice_2StridedSlicelstm_40/transpose:y:0&lstm_40/strided_slice_2/stack:output:0(lstm_40/strided_slice_2/stack_1:output:0(lstm_40/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
*lstm_40/lstm_cell_40/MatMul/ReadVariableOpReadVariableOp3lstm_40_lstm_cell_40_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
lstm_40/lstm_cell_40/MatMulMatMul lstm_40/strided_slice_2:output:02lstm_40/lstm_cell_40/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
,lstm_40/lstm_cell_40/MatMul_1/ReadVariableOpReadVariableOp5lstm_40_lstm_cell_40_matmul_1_readvariableop_resource*
_output_shapes
:	d�*
dtype0�
lstm_40/lstm_cell_40/MatMul_1MatMullstm_40/zeros:output:04lstm_40/lstm_cell_40/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
lstm_40/lstm_cell_40/addAddV2%lstm_40/lstm_cell_40/MatMul:product:0'lstm_40/lstm_cell_40/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
+lstm_40/lstm_cell_40/BiasAdd/ReadVariableOpReadVariableOp4lstm_40_lstm_cell_40_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
lstm_40/lstm_cell_40/BiasAddBiasAddlstm_40/lstm_cell_40/add:z:03lstm_40/lstm_cell_40/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������f
$lstm_40/lstm_cell_40/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_40/lstm_cell_40/splitSplit-lstm_40/lstm_cell_40/split/split_dim:output:0%lstm_40/lstm_cell_40/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������d:���������d:���������d:���������d*
	num_split~
lstm_40/lstm_cell_40/SigmoidSigmoid#lstm_40/lstm_cell_40/split:output:0*
T0*'
_output_shapes
:���������d�
lstm_40/lstm_cell_40/Sigmoid_1Sigmoid#lstm_40/lstm_cell_40/split:output:1*
T0*'
_output_shapes
:���������d�
lstm_40/lstm_cell_40/mulMul"lstm_40/lstm_cell_40/Sigmoid_1:y:0lstm_40/zeros_1:output:0*
T0*'
_output_shapes
:���������dx
lstm_40/lstm_cell_40/ReluRelu#lstm_40/lstm_cell_40/split:output:2*
T0*'
_output_shapes
:���������d�
lstm_40/lstm_cell_40/mul_1Mul lstm_40/lstm_cell_40/Sigmoid:y:0'lstm_40/lstm_cell_40/Relu:activations:0*
T0*'
_output_shapes
:���������d�
lstm_40/lstm_cell_40/add_1AddV2lstm_40/lstm_cell_40/mul:z:0lstm_40/lstm_cell_40/mul_1:z:0*
T0*'
_output_shapes
:���������d�
lstm_40/lstm_cell_40/Sigmoid_2Sigmoid#lstm_40/lstm_cell_40/split:output:3*
T0*'
_output_shapes
:���������du
lstm_40/lstm_cell_40/Relu_1Relulstm_40/lstm_cell_40/add_1:z:0*
T0*'
_output_shapes
:���������d�
lstm_40/lstm_cell_40/mul_2Mul"lstm_40/lstm_cell_40/Sigmoid_2:y:0)lstm_40/lstm_cell_40/Relu_1:activations:0*
T0*'
_output_shapes
:���������dv
%lstm_40/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   f
$lstm_40/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_40/TensorArrayV2_1TensorListReserve.lstm_40/TensorArrayV2_1/element_shape:output:0-lstm_40/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���N
lstm_40/timeConst*
_output_shapes
: *
dtype0*
value	B : k
 lstm_40/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������\
lstm_40/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
lstm_40/whileWhile#lstm_40/while/loop_counter:output:0)lstm_40/while/maximum_iterations:output:0lstm_40/time:output:0 lstm_40/TensorArrayV2_1:handle:0lstm_40/zeros:output:0lstm_40/zeros_1:output:0 lstm_40/strided_slice_1:output:0?lstm_40/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_40_lstm_cell_40_matmul_readvariableop_resource5lstm_40_lstm_cell_40_matmul_1_readvariableop_resource4lstm_40_lstm_cell_40_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������d:���������d: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *&
bodyR
lstm_40_while_body_1378556*&
condR
lstm_40_while_cond_1378555*K
output_shapes:
8: : : : :���������d:���������d: : : : : *
parallel_iterations �
8lstm_40/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   �
*lstm_40/TensorArrayV2Stack/TensorListStackTensorListStacklstm_40/while:output:3Alstm_40/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������d*
element_dtype0*
num_elementsp
lstm_40/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������i
lstm_40/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: i
lstm_40/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
lstm_40/strided_slice_3StridedSlice3lstm_40/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_40/strided_slice_3/stack:output:0(lstm_40/strided_slice_3/stack_1:output:0(lstm_40/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������d*
shrink_axis_maskm
lstm_40/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
lstm_40/transpose_1	Transpose3lstm_40/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_40/transpose_1/perm:output:0*
T0*+
_output_shapes
:���������dc
lstm_40/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    �
dense_120/MatMul/ReadVariableOpReadVariableOp(dense_120_matmul_readvariableop_resource*
_output_shapes

:d2*
dtype0�
dense_120/MatMulMatMul lstm_40/strided_slice_3:output:0'dense_120/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2�
 dense_120/BiasAdd/ReadVariableOpReadVariableOp)dense_120_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0�
dense_120/BiasAddBiasAdddense_120/MatMul:product:0(dense_120/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2d
dense_120/ReluReludense_120/BiasAdd:output:0*
T0*'
_output_shapes
:���������2�
dense_121/MatMul/ReadVariableOpReadVariableOp(dense_121_matmul_readvariableop_resource*
_output_shapes

:22*
dtype0�
dense_121/MatMulMatMuldense_120/Relu:activations:0'dense_121/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2�
 dense_121/BiasAdd/ReadVariableOpReadVariableOp)dense_121_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0�
dense_121/BiasAddBiasAdddense_121/MatMul:product:0(dense_121/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2d
dense_121/ReluReludense_121/BiasAdd:output:0*
T0*'
_output_shapes
:���������2�
dense_122/MatMul/ReadVariableOpReadVariableOp(dense_122_matmul_readvariableop_resource*
_output_shapes

:2*
dtype0�
dense_122/MatMulMatMuldense_121/Relu:activations:0'dense_122/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
 dense_122/BiasAdd/ReadVariableOpReadVariableOp)dense_122_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_122/BiasAddBiasAdddense_122/MatMul:product:0(dense_122/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������i
IdentityIdentitydense_122/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp!^dense_120/BiasAdd/ReadVariableOp ^dense_120/MatMul/ReadVariableOp!^dense_121/BiasAdd/ReadVariableOp ^dense_121/MatMul/ReadVariableOp!^dense_122/BiasAdd/ReadVariableOp ^dense_122/MatMul/ReadVariableOp,^lstm_40/lstm_cell_40/BiasAdd/ReadVariableOp+^lstm_40/lstm_cell_40/MatMul/ReadVariableOp-^lstm_40/lstm_cell_40/MatMul_1/ReadVariableOp^lstm_40/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):���������: : : : : : : : : 2D
 dense_120/BiasAdd/ReadVariableOp dense_120/BiasAdd/ReadVariableOp2B
dense_120/MatMul/ReadVariableOpdense_120/MatMul/ReadVariableOp2D
 dense_121/BiasAdd/ReadVariableOp dense_121/BiasAdd/ReadVariableOp2B
dense_121/MatMul/ReadVariableOpdense_121/MatMul/ReadVariableOp2D
 dense_122/BiasAdd/ReadVariableOp dense_122/BiasAdd/ReadVariableOp2B
dense_122/MatMul/ReadVariableOpdense_122/MatMul/ReadVariableOp2Z
+lstm_40/lstm_cell_40/BiasAdd/ReadVariableOp+lstm_40/lstm_cell_40/BiasAdd/ReadVariableOp2X
*lstm_40/lstm_cell_40/MatMul/ReadVariableOp*lstm_40/lstm_cell_40/MatMul/ReadVariableOp2\
,lstm_40/lstm_cell_40/MatMul_1/ReadVariableOp,lstm_40/lstm_cell_40/MatMul_1/ReadVariableOp2
lstm_40/whilelstm_40/while:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�	
"__inference__wrapped_model_1377325
lstm_40_inputT
Asequential_40_lstm_40_lstm_cell_40_matmul_readvariableop_resource:	�V
Csequential_40_lstm_40_lstm_cell_40_matmul_1_readvariableop_resource:	d�Q
Bsequential_40_lstm_40_lstm_cell_40_biasadd_readvariableop_resource:	�H
6sequential_40_dense_120_matmul_readvariableop_resource:d2E
7sequential_40_dense_120_biasadd_readvariableop_resource:2H
6sequential_40_dense_121_matmul_readvariableop_resource:22E
7sequential_40_dense_121_biasadd_readvariableop_resource:2H
6sequential_40_dense_122_matmul_readvariableop_resource:2E
7sequential_40_dense_122_biasadd_readvariableop_resource:
identity��.sequential_40/dense_120/BiasAdd/ReadVariableOp�-sequential_40/dense_120/MatMul/ReadVariableOp�.sequential_40/dense_121/BiasAdd/ReadVariableOp�-sequential_40/dense_121/MatMul/ReadVariableOp�.sequential_40/dense_122/BiasAdd/ReadVariableOp�-sequential_40/dense_122/MatMul/ReadVariableOp�9sequential_40/lstm_40/lstm_cell_40/BiasAdd/ReadVariableOp�8sequential_40/lstm_40/lstm_cell_40/MatMul/ReadVariableOp�:sequential_40/lstm_40/lstm_cell_40/MatMul_1/ReadVariableOp�sequential_40/lstm_40/whileX
sequential_40/lstm_40/ShapeShapelstm_40_input*
T0*
_output_shapes
:s
)sequential_40/lstm_40/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+sequential_40/lstm_40/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+sequential_40/lstm_40/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
#sequential_40/lstm_40/strided_sliceStridedSlice$sequential_40/lstm_40/Shape:output:02sequential_40/lstm_40/strided_slice/stack:output:04sequential_40/lstm_40/strided_slice/stack_1:output:04sequential_40/lstm_40/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
$sequential_40/lstm_40/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d�
"sequential_40/lstm_40/zeros/packedPack,sequential_40/lstm_40/strided_slice:output:0-sequential_40/lstm_40/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:f
!sequential_40/lstm_40/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
sequential_40/lstm_40/zerosFill+sequential_40/lstm_40/zeros/packed:output:0*sequential_40/lstm_40/zeros/Const:output:0*
T0*'
_output_shapes
:���������dh
&sequential_40/lstm_40/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d�
$sequential_40/lstm_40/zeros_1/packedPack,sequential_40/lstm_40/strided_slice:output:0/sequential_40/lstm_40/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:h
#sequential_40/lstm_40/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
sequential_40/lstm_40/zeros_1Fill-sequential_40/lstm_40/zeros_1/packed:output:0,sequential_40/lstm_40/zeros_1/Const:output:0*
T0*'
_output_shapes
:���������dy
$sequential_40/lstm_40/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
sequential_40/lstm_40/transpose	Transposelstm_40_input-sequential_40/lstm_40/transpose/perm:output:0*
T0*+
_output_shapes
:���������p
sequential_40/lstm_40/Shape_1Shape#sequential_40/lstm_40/transpose:y:0*
T0*
_output_shapes
:u
+sequential_40/lstm_40/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-sequential_40/lstm_40/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-sequential_40/lstm_40/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
%sequential_40/lstm_40/strided_slice_1StridedSlice&sequential_40/lstm_40/Shape_1:output:04sequential_40/lstm_40/strided_slice_1/stack:output:06sequential_40/lstm_40/strided_slice_1/stack_1:output:06sequential_40/lstm_40/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask|
1sequential_40/lstm_40/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
#sequential_40/lstm_40/TensorArrayV2TensorListReserve:sequential_40/lstm_40/TensorArrayV2/element_shape:output:0.sequential_40/lstm_40/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
Ksequential_40/lstm_40/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
=sequential_40/lstm_40/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor#sequential_40/lstm_40/transpose:y:0Tsequential_40/lstm_40/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���u
+sequential_40/lstm_40/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-sequential_40/lstm_40/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-sequential_40/lstm_40/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
%sequential_40/lstm_40/strided_slice_2StridedSlice#sequential_40/lstm_40/transpose:y:04sequential_40/lstm_40/strided_slice_2/stack:output:06sequential_40/lstm_40/strided_slice_2/stack_1:output:06sequential_40/lstm_40/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
8sequential_40/lstm_40/lstm_cell_40/MatMul/ReadVariableOpReadVariableOpAsequential_40_lstm_40_lstm_cell_40_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
)sequential_40/lstm_40/lstm_cell_40/MatMulMatMul.sequential_40/lstm_40/strided_slice_2:output:0@sequential_40/lstm_40/lstm_cell_40/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
:sequential_40/lstm_40/lstm_cell_40/MatMul_1/ReadVariableOpReadVariableOpCsequential_40_lstm_40_lstm_cell_40_matmul_1_readvariableop_resource*
_output_shapes
:	d�*
dtype0�
+sequential_40/lstm_40/lstm_cell_40/MatMul_1MatMul$sequential_40/lstm_40/zeros:output:0Bsequential_40/lstm_40/lstm_cell_40/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
&sequential_40/lstm_40/lstm_cell_40/addAddV23sequential_40/lstm_40/lstm_cell_40/MatMul:product:05sequential_40/lstm_40/lstm_cell_40/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
9sequential_40/lstm_40/lstm_cell_40/BiasAdd/ReadVariableOpReadVariableOpBsequential_40_lstm_40_lstm_cell_40_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
*sequential_40/lstm_40/lstm_cell_40/BiasAddBiasAdd*sequential_40/lstm_40/lstm_cell_40/add:z:0Asequential_40/lstm_40/lstm_cell_40/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������t
2sequential_40/lstm_40/lstm_cell_40/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
(sequential_40/lstm_40/lstm_cell_40/splitSplit;sequential_40/lstm_40/lstm_cell_40/split/split_dim:output:03sequential_40/lstm_40/lstm_cell_40/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������d:���������d:���������d:���������d*
	num_split�
*sequential_40/lstm_40/lstm_cell_40/SigmoidSigmoid1sequential_40/lstm_40/lstm_cell_40/split:output:0*
T0*'
_output_shapes
:���������d�
,sequential_40/lstm_40/lstm_cell_40/Sigmoid_1Sigmoid1sequential_40/lstm_40/lstm_cell_40/split:output:1*
T0*'
_output_shapes
:���������d�
&sequential_40/lstm_40/lstm_cell_40/mulMul0sequential_40/lstm_40/lstm_cell_40/Sigmoid_1:y:0&sequential_40/lstm_40/zeros_1:output:0*
T0*'
_output_shapes
:���������d�
'sequential_40/lstm_40/lstm_cell_40/ReluRelu1sequential_40/lstm_40/lstm_cell_40/split:output:2*
T0*'
_output_shapes
:���������d�
(sequential_40/lstm_40/lstm_cell_40/mul_1Mul.sequential_40/lstm_40/lstm_cell_40/Sigmoid:y:05sequential_40/lstm_40/lstm_cell_40/Relu:activations:0*
T0*'
_output_shapes
:���������d�
(sequential_40/lstm_40/lstm_cell_40/add_1AddV2*sequential_40/lstm_40/lstm_cell_40/mul:z:0,sequential_40/lstm_40/lstm_cell_40/mul_1:z:0*
T0*'
_output_shapes
:���������d�
,sequential_40/lstm_40/lstm_cell_40/Sigmoid_2Sigmoid1sequential_40/lstm_40/lstm_cell_40/split:output:3*
T0*'
_output_shapes
:���������d�
)sequential_40/lstm_40/lstm_cell_40/Relu_1Relu,sequential_40/lstm_40/lstm_cell_40/add_1:z:0*
T0*'
_output_shapes
:���������d�
(sequential_40/lstm_40/lstm_cell_40/mul_2Mul0sequential_40/lstm_40/lstm_cell_40/Sigmoid_2:y:07sequential_40/lstm_40/lstm_cell_40/Relu_1:activations:0*
T0*'
_output_shapes
:���������d�
3sequential_40/lstm_40/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   t
2sequential_40/lstm_40/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :�
%sequential_40/lstm_40/TensorArrayV2_1TensorListReserve<sequential_40/lstm_40/TensorArrayV2_1/element_shape:output:0;sequential_40/lstm_40/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���\
sequential_40/lstm_40/timeConst*
_output_shapes
: *
dtype0*
value	B : y
.sequential_40/lstm_40/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������j
(sequential_40/lstm_40/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
sequential_40/lstm_40/whileWhile1sequential_40/lstm_40/while/loop_counter:output:07sequential_40/lstm_40/while/maximum_iterations:output:0#sequential_40/lstm_40/time:output:0.sequential_40/lstm_40/TensorArrayV2_1:handle:0$sequential_40/lstm_40/zeros:output:0&sequential_40/lstm_40/zeros_1:output:0.sequential_40/lstm_40/strided_slice_1:output:0Msequential_40/lstm_40/TensorArrayUnstack/TensorListFromTensor:output_handle:0Asequential_40_lstm_40_lstm_cell_40_matmul_readvariableop_resourceCsequential_40_lstm_40_lstm_cell_40_matmul_1_readvariableop_resourceBsequential_40_lstm_40_lstm_cell_40_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������d:���������d: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *4
body,R*
(sequential_40_lstm_40_while_body_1377220*4
cond,R*
(sequential_40_lstm_40_while_cond_1377219*K
output_shapes:
8: : : : :���������d:���������d: : : : : *
parallel_iterations �
Fsequential_40/lstm_40/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   �
8sequential_40/lstm_40/TensorArrayV2Stack/TensorListStackTensorListStack$sequential_40/lstm_40/while:output:3Osequential_40/lstm_40/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������d*
element_dtype0*
num_elements~
+sequential_40/lstm_40/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������w
-sequential_40/lstm_40/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: w
-sequential_40/lstm_40/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
%sequential_40/lstm_40/strided_slice_3StridedSliceAsequential_40/lstm_40/TensorArrayV2Stack/TensorListStack:tensor:04sequential_40/lstm_40/strided_slice_3/stack:output:06sequential_40/lstm_40/strided_slice_3/stack_1:output:06sequential_40/lstm_40/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������d*
shrink_axis_mask{
&sequential_40/lstm_40/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
!sequential_40/lstm_40/transpose_1	TransposeAsequential_40/lstm_40/TensorArrayV2Stack/TensorListStack:tensor:0/sequential_40/lstm_40/transpose_1/perm:output:0*
T0*+
_output_shapes
:���������dq
sequential_40/lstm_40/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    �
-sequential_40/dense_120/MatMul/ReadVariableOpReadVariableOp6sequential_40_dense_120_matmul_readvariableop_resource*
_output_shapes

:d2*
dtype0�
sequential_40/dense_120/MatMulMatMul.sequential_40/lstm_40/strided_slice_3:output:05sequential_40/dense_120/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2�
.sequential_40/dense_120/BiasAdd/ReadVariableOpReadVariableOp7sequential_40_dense_120_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0�
sequential_40/dense_120/BiasAddBiasAdd(sequential_40/dense_120/MatMul:product:06sequential_40/dense_120/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2�
sequential_40/dense_120/ReluRelu(sequential_40/dense_120/BiasAdd:output:0*
T0*'
_output_shapes
:���������2�
-sequential_40/dense_121/MatMul/ReadVariableOpReadVariableOp6sequential_40_dense_121_matmul_readvariableop_resource*
_output_shapes

:22*
dtype0�
sequential_40/dense_121/MatMulMatMul*sequential_40/dense_120/Relu:activations:05sequential_40/dense_121/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2�
.sequential_40/dense_121/BiasAdd/ReadVariableOpReadVariableOp7sequential_40_dense_121_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0�
sequential_40/dense_121/BiasAddBiasAdd(sequential_40/dense_121/MatMul:product:06sequential_40/dense_121/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2�
sequential_40/dense_121/ReluRelu(sequential_40/dense_121/BiasAdd:output:0*
T0*'
_output_shapes
:���������2�
-sequential_40/dense_122/MatMul/ReadVariableOpReadVariableOp6sequential_40_dense_122_matmul_readvariableop_resource*
_output_shapes

:2*
dtype0�
sequential_40/dense_122/MatMulMatMul*sequential_40/dense_121/Relu:activations:05sequential_40/dense_122/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
.sequential_40/dense_122/BiasAdd/ReadVariableOpReadVariableOp7sequential_40_dense_122_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
sequential_40/dense_122/BiasAddBiasAdd(sequential_40/dense_122/MatMul:product:06sequential_40/dense_122/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������w
IdentityIdentity(sequential_40/dense_122/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp/^sequential_40/dense_120/BiasAdd/ReadVariableOp.^sequential_40/dense_120/MatMul/ReadVariableOp/^sequential_40/dense_121/BiasAdd/ReadVariableOp.^sequential_40/dense_121/MatMul/ReadVariableOp/^sequential_40/dense_122/BiasAdd/ReadVariableOp.^sequential_40/dense_122/MatMul/ReadVariableOp:^sequential_40/lstm_40/lstm_cell_40/BiasAdd/ReadVariableOp9^sequential_40/lstm_40/lstm_cell_40/MatMul/ReadVariableOp;^sequential_40/lstm_40/lstm_cell_40/MatMul_1/ReadVariableOp^sequential_40/lstm_40/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):���������: : : : : : : : : 2`
.sequential_40/dense_120/BiasAdd/ReadVariableOp.sequential_40/dense_120/BiasAdd/ReadVariableOp2^
-sequential_40/dense_120/MatMul/ReadVariableOp-sequential_40/dense_120/MatMul/ReadVariableOp2`
.sequential_40/dense_121/BiasAdd/ReadVariableOp.sequential_40/dense_121/BiasAdd/ReadVariableOp2^
-sequential_40/dense_121/MatMul/ReadVariableOp-sequential_40/dense_121/MatMul/ReadVariableOp2`
.sequential_40/dense_122/BiasAdd/ReadVariableOp.sequential_40/dense_122/BiasAdd/ReadVariableOp2^
-sequential_40/dense_122/MatMul/ReadVariableOp-sequential_40/dense_122/MatMul/ReadVariableOp2v
9sequential_40/lstm_40/lstm_cell_40/BiasAdd/ReadVariableOp9sequential_40/lstm_40/lstm_cell_40/BiasAdd/ReadVariableOp2t
8sequential_40/lstm_40/lstm_cell_40/MatMul/ReadVariableOp8sequential_40/lstm_40/lstm_cell_40/MatMul/ReadVariableOp2x
:sequential_40/lstm_40/lstm_cell_40/MatMul_1/ReadVariableOp:sequential_40/lstm_40/lstm_cell_40/MatMul_1/ReadVariableOp2:
sequential_40/lstm_40/whilesequential_40/lstm_40/while:Z V
+
_output_shapes
:���������
'
_user_specified_namelstm_40_input
�

�
F__inference_dense_121_layer_call_and_return_conditional_losses_1377866

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
�9
�
while_body_1378765
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_40_matmul_readvariableop_resource_0:	�H
5while_lstm_cell_40_matmul_1_readvariableop_resource_0:	d�C
4while_lstm_cell_40_biasadd_readvariableop_resource_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_40_matmul_readvariableop_resource:	�F
3while_lstm_cell_40_matmul_1_readvariableop_resource:	d�A
2while_lstm_cell_40_biasadd_readvariableop_resource:	���)while/lstm_cell_40/BiasAdd/ReadVariableOp�(while/lstm_cell_40/MatMul/ReadVariableOp�*while/lstm_cell_40/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
(while/lstm_cell_40/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_40_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
while/lstm_cell_40/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_40/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
*while/lstm_cell_40/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_40_matmul_1_readvariableop_resource_0*
_output_shapes
:	d�*
dtype0�
while/lstm_cell_40/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_40/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
while/lstm_cell_40/addAddV2#while/lstm_cell_40/MatMul:product:0%while/lstm_cell_40/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
)while/lstm_cell_40/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_40_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype0�
while/lstm_cell_40/BiasAddBiasAddwhile/lstm_cell_40/add:z:01while/lstm_cell_40/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������d
"while/lstm_cell_40/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/lstm_cell_40/splitSplit+while/lstm_cell_40/split/split_dim:output:0#while/lstm_cell_40/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������d:���������d:���������d:���������d*
	num_splitz
while/lstm_cell_40/SigmoidSigmoid!while/lstm_cell_40/split:output:0*
T0*'
_output_shapes
:���������d|
while/lstm_cell_40/Sigmoid_1Sigmoid!while/lstm_cell_40/split:output:1*
T0*'
_output_shapes
:���������d�
while/lstm_cell_40/mulMul while/lstm_cell_40/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������dt
while/lstm_cell_40/ReluRelu!while/lstm_cell_40/split:output:2*
T0*'
_output_shapes
:���������d�
while/lstm_cell_40/mul_1Mulwhile/lstm_cell_40/Sigmoid:y:0%while/lstm_cell_40/Relu:activations:0*
T0*'
_output_shapes
:���������d�
while/lstm_cell_40/add_1AddV2while/lstm_cell_40/mul:z:0while/lstm_cell_40/mul_1:z:0*
T0*'
_output_shapes
:���������d|
while/lstm_cell_40/Sigmoid_2Sigmoid!while/lstm_cell_40/split:output:3*
T0*'
_output_shapes
:���������dq
while/lstm_cell_40/Relu_1Reluwhile/lstm_cell_40/add_1:z:0*
T0*'
_output_shapes
:���������d�
while/lstm_cell_40/mul_2Mul while/lstm_cell_40/Sigmoid_2:y:0'while/lstm_cell_40/Relu_1:activations:0*
T0*'
_output_shapes
:���������dr
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : �
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell_40/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_40/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:���������dy
while/Identity_5Identitywhile/lstm_cell_40/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:���������d�

while/NoOpNoOp*^while/lstm_cell_40/BiasAdd/ReadVariableOp)^while/lstm_cell_40/MatMul/ReadVariableOp+^while/lstm_cell_40/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_40_biasadd_readvariableop_resource4while_lstm_cell_40_biasadd_readvariableop_resource_0"l
3while_lstm_cell_40_matmul_1_readvariableop_resource5while_lstm_cell_40_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_40_matmul_readvariableop_resource3while_lstm_cell_40_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������d:���������d: : : : : 2V
)while/lstm_cell_40/BiasAdd/ReadVariableOp)while/lstm_cell_40/BiasAdd/ReadVariableOp2T
(while/lstm_cell_40/MatMul/ReadVariableOp(while/lstm_cell_40/MatMul/ReadVariableOp2X
*while/lstm_cell_40/MatMul_1/ReadVariableOp*while/lstm_cell_40/MatMul_1/ReadVariableOp: 
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
while_cond_1378012
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1378012___redundant_placeholder05
1while_while_cond_1378012___redundant_placeholder15
1while_while_cond_1378012___redundant_placeholder25
1while_while_cond_1378012___redundant_placeholder3
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

�
F__inference_dense_120_layer_call_and_return_conditional_losses_1377849

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
#__inference__traced_restore_1379691
file_prefix3
!assignvariableop_dense_120_kernel:d2/
!assignvariableop_1_dense_120_bias:25
#assignvariableop_2_dense_121_kernel:22/
!assignvariableop_3_dense_121_bias:25
#assignvariableop_4_dense_122_kernel:2/
!assignvariableop_5_dense_122_bias:A
.assignvariableop_6_lstm_40_lstm_cell_40_kernel:	�K
8assignvariableop_7_lstm_40_lstm_cell_40_recurrent_kernel:	d�;
,assignvariableop_8_lstm_40_lstm_cell_40_bias:	�&
assignvariableop_9_adam_iter:	 )
assignvariableop_10_adam_beta_1: )
assignvariableop_11_adam_beta_2: (
assignvariableop_12_adam_decay: 0
&assignvariableop_13_adam_learning_rate: %
assignvariableop_14_total_1: %
assignvariableop_15_count_1: #
assignvariableop_16_total: #
assignvariableop_17_count: =
+assignvariableop_18_adam_dense_120_kernel_m:d27
)assignvariableop_19_adam_dense_120_bias_m:2=
+assignvariableop_20_adam_dense_121_kernel_m:227
)assignvariableop_21_adam_dense_121_bias_m:2=
+assignvariableop_22_adam_dense_122_kernel_m:27
)assignvariableop_23_adam_dense_122_bias_m:I
6assignvariableop_24_adam_lstm_40_lstm_cell_40_kernel_m:	�S
@assignvariableop_25_adam_lstm_40_lstm_cell_40_recurrent_kernel_m:	d�C
4assignvariableop_26_adam_lstm_40_lstm_cell_40_bias_m:	�=
+assignvariableop_27_adam_dense_120_kernel_v:d27
)assignvariableop_28_adam_dense_120_bias_v:2=
+assignvariableop_29_adam_dense_121_kernel_v:227
)assignvariableop_30_adam_dense_121_bias_v:2=
+assignvariableop_31_adam_dense_122_kernel_v:27
)assignvariableop_32_adam_dense_122_bias_v:I
6assignvariableop_33_adam_lstm_40_lstm_cell_40_kernel_v:	�S
@assignvariableop_34_adam_lstm_40_lstm_cell_40_recurrent_kernel_v:	d�C
4assignvariableop_35_adam_lstm_40_lstm_cell_40_bias_v:	�
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
AssignVariableOpAssignVariableOp!assignvariableop_dense_120_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp!assignvariableop_1_dense_120_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp#assignvariableop_2_dense_121_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp!assignvariableop_3_dense_121_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp#assignvariableop_4_dense_122_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp!assignvariableop_5_dense_122_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp.assignvariableop_6_lstm_40_lstm_cell_40_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp8assignvariableop_7_lstm_40_lstm_cell_40_recurrent_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp,assignvariableop_8_lstm_40_lstm_cell_40_biasIdentity_8:output:0"/device:CPU:0*
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
AssignVariableOp_18AssignVariableOp+assignvariableop_18_adam_dense_120_kernel_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp)assignvariableop_19_adam_dense_120_bias_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOp+assignvariableop_20_adam_dense_121_kernel_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOp)assignvariableop_21_adam_dense_121_bias_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOp+assignvariableop_22_adam_dense_122_kernel_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOp)assignvariableop_23_adam_dense_122_bias_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOp6assignvariableop_24_adam_lstm_40_lstm_cell_40_kernel_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOp@assignvariableop_25_adam_lstm_40_lstm_cell_40_recurrent_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOp4assignvariableop_26_adam_lstm_40_lstm_cell_40_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOp+assignvariableop_27_adam_dense_120_kernel_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOp)assignvariableop_28_adam_dense_120_bias_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOp+assignvariableop_29_adam_dense_121_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_30AssignVariableOp)assignvariableop_30_adam_dense_121_bias_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_31AssignVariableOp+assignvariableop_31_adam_dense_122_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_32AssignVariableOp)assignvariableop_32_adam_dense_122_bias_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_33AssignVariableOp6assignvariableop_33_adam_lstm_40_lstm_cell_40_kernel_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_34AssignVariableOp@assignvariableop_34_adam_lstm_40_lstm_cell_40_recurrent_kernel_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_35AssignVariableOp4assignvariableop_35_adam_lstm_40_lstm_cell_40_bias_vIdentity_35:output:0"/device:CPU:0*
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
while_cond_1377406
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1377406___redundant_placeholder05
1while_while_cond_1377406___redundant_placeholder15
1while_while_cond_1377406___redundant_placeholder25
1while_while_cond_1377406___redundant_placeholder3
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
/__inference_sequential_40_layer_call_fn_1378202
lstm_40_input
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
StatefulPartitionedCallStatefulPartitionedCalllstm_40_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
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
GPU 2J 8� *S
fNRL
J__inference_sequential_40_layer_call_and_return_conditional_losses_1378158o
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
_user_specified_namelstm_40_input
�
�
.__inference_lstm_cell_40_layer_call_fn_1379361

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
GPU 2J 8� *R
fMRK
I__inference_lstm_cell_40_layer_call_and_return_conditional_losses_1377392o
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
�K
�
D__inference_lstm_40_layer_call_and_return_conditional_losses_1378995
inputs_0>
+lstm_cell_40_matmul_readvariableop_resource:	�@
-lstm_cell_40_matmul_1_readvariableop_resource:	d�;
,lstm_cell_40_biasadd_readvariableop_resource:	�
identity��#lstm_cell_40/BiasAdd/ReadVariableOp�"lstm_cell_40/MatMul/ReadVariableOp�$lstm_cell_40/MatMul_1/ReadVariableOp�while=
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
"lstm_cell_40/MatMul/ReadVariableOpReadVariableOp+lstm_cell_40_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
lstm_cell_40/MatMulMatMulstrided_slice_2:output:0*lstm_cell_40/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
$lstm_cell_40/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_40_matmul_1_readvariableop_resource*
_output_shapes
:	d�*
dtype0�
lstm_cell_40/MatMul_1MatMulzeros:output:0,lstm_cell_40/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
lstm_cell_40/addAddV2lstm_cell_40/MatMul:product:0lstm_cell_40/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
#lstm_cell_40/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_40_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
lstm_cell_40/BiasAddBiasAddlstm_cell_40/add:z:0+lstm_cell_40/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������^
lstm_cell_40/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_cell_40/splitSplit%lstm_cell_40/split/split_dim:output:0lstm_cell_40/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������d:���������d:���������d:���������d*
	num_splitn
lstm_cell_40/SigmoidSigmoidlstm_cell_40/split:output:0*
T0*'
_output_shapes
:���������dp
lstm_cell_40/Sigmoid_1Sigmoidlstm_cell_40/split:output:1*
T0*'
_output_shapes
:���������dw
lstm_cell_40/mulMullstm_cell_40/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������dh
lstm_cell_40/ReluRelulstm_cell_40/split:output:2*
T0*'
_output_shapes
:���������d�
lstm_cell_40/mul_1Mullstm_cell_40/Sigmoid:y:0lstm_cell_40/Relu:activations:0*
T0*'
_output_shapes
:���������d{
lstm_cell_40/add_1AddV2lstm_cell_40/mul:z:0lstm_cell_40/mul_1:z:0*
T0*'
_output_shapes
:���������dp
lstm_cell_40/Sigmoid_2Sigmoidlstm_cell_40/split:output:3*
T0*'
_output_shapes
:���������de
lstm_cell_40/Relu_1Relulstm_cell_40/add_1:z:0*
T0*'
_output_shapes
:���������d�
lstm_cell_40/mul_2Mullstm_cell_40/Sigmoid_2:y:0!lstm_cell_40/Relu_1:activations:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_40_matmul_readvariableop_resource-lstm_cell_40_matmul_1_readvariableop_resource,lstm_cell_40_biasadd_readvariableop_resource*
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
bodyR
while_body_1378910*
condR
while_cond_1378909*K
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
NoOpNoOp$^lstm_cell_40/BiasAdd/ReadVariableOp#^lstm_cell_40/MatMul/ReadVariableOp%^lstm_cell_40/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 2J
#lstm_cell_40/BiasAdd/ReadVariableOp#lstm_cell_40/BiasAdd/ReadVariableOp2H
"lstm_cell_40/MatMul/ReadVariableOp"lstm_cell_40/MatMul/ReadVariableOp2L
$lstm_cell_40/MatMul_1/ReadVariableOp$lstm_cell_40/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :������������������
"
_user_specified_name
inputs/0
�m
�
J__inference_sequential_40_layer_call_and_return_conditional_losses_1378496

inputsF
3lstm_40_lstm_cell_40_matmul_readvariableop_resource:	�H
5lstm_40_lstm_cell_40_matmul_1_readvariableop_resource:	d�C
4lstm_40_lstm_cell_40_biasadd_readvariableop_resource:	�:
(dense_120_matmul_readvariableop_resource:d27
)dense_120_biasadd_readvariableop_resource:2:
(dense_121_matmul_readvariableop_resource:227
)dense_121_biasadd_readvariableop_resource:2:
(dense_122_matmul_readvariableop_resource:27
)dense_122_biasadd_readvariableop_resource:
identity�� dense_120/BiasAdd/ReadVariableOp�dense_120/MatMul/ReadVariableOp� dense_121/BiasAdd/ReadVariableOp�dense_121/MatMul/ReadVariableOp� dense_122/BiasAdd/ReadVariableOp�dense_122/MatMul/ReadVariableOp�+lstm_40/lstm_cell_40/BiasAdd/ReadVariableOp�*lstm_40/lstm_cell_40/MatMul/ReadVariableOp�,lstm_40/lstm_cell_40/MatMul_1/ReadVariableOp�lstm_40/whileC
lstm_40/ShapeShapeinputs*
T0*
_output_shapes
:e
lstm_40/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
lstm_40/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
lstm_40/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
lstm_40/strided_sliceStridedSlicelstm_40/Shape:output:0$lstm_40/strided_slice/stack:output:0&lstm_40/strided_slice/stack_1:output:0&lstm_40/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
lstm_40/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d�
lstm_40/zeros/packedPacklstm_40/strided_slice:output:0lstm_40/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:X
lstm_40/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
lstm_40/zerosFilllstm_40/zeros/packed:output:0lstm_40/zeros/Const:output:0*
T0*'
_output_shapes
:���������dZ
lstm_40/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d�
lstm_40/zeros_1/packedPacklstm_40/strided_slice:output:0!lstm_40/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:Z
lstm_40/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
lstm_40/zeros_1Filllstm_40/zeros_1/packed:output:0lstm_40/zeros_1/Const:output:0*
T0*'
_output_shapes
:���������dk
lstm_40/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          }
lstm_40/transpose	Transposeinputslstm_40/transpose/perm:output:0*
T0*+
_output_shapes
:���������T
lstm_40/Shape_1Shapelstm_40/transpose:y:0*
T0*
_output_shapes
:g
lstm_40/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: i
lstm_40/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
lstm_40/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
lstm_40/strided_slice_1StridedSlicelstm_40/Shape_1:output:0&lstm_40/strided_slice_1/stack:output:0(lstm_40/strided_slice_1/stack_1:output:0(lstm_40/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskn
#lstm_40/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
lstm_40/TensorArrayV2TensorListReserve,lstm_40/TensorArrayV2/element_shape:output:0 lstm_40/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
=lstm_40/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
/lstm_40/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_40/transpose:y:0Flstm_40/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���g
lstm_40/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: i
lstm_40/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
lstm_40/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
lstm_40/strided_slice_2StridedSlicelstm_40/transpose:y:0&lstm_40/strided_slice_2/stack:output:0(lstm_40/strided_slice_2/stack_1:output:0(lstm_40/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
*lstm_40/lstm_cell_40/MatMul/ReadVariableOpReadVariableOp3lstm_40_lstm_cell_40_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
lstm_40/lstm_cell_40/MatMulMatMul lstm_40/strided_slice_2:output:02lstm_40/lstm_cell_40/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
,lstm_40/lstm_cell_40/MatMul_1/ReadVariableOpReadVariableOp5lstm_40_lstm_cell_40_matmul_1_readvariableop_resource*
_output_shapes
:	d�*
dtype0�
lstm_40/lstm_cell_40/MatMul_1MatMullstm_40/zeros:output:04lstm_40/lstm_cell_40/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
lstm_40/lstm_cell_40/addAddV2%lstm_40/lstm_cell_40/MatMul:product:0'lstm_40/lstm_cell_40/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
+lstm_40/lstm_cell_40/BiasAdd/ReadVariableOpReadVariableOp4lstm_40_lstm_cell_40_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
lstm_40/lstm_cell_40/BiasAddBiasAddlstm_40/lstm_cell_40/add:z:03lstm_40/lstm_cell_40/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������f
$lstm_40/lstm_cell_40/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_40/lstm_cell_40/splitSplit-lstm_40/lstm_cell_40/split/split_dim:output:0%lstm_40/lstm_cell_40/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������d:���������d:���������d:���������d*
	num_split~
lstm_40/lstm_cell_40/SigmoidSigmoid#lstm_40/lstm_cell_40/split:output:0*
T0*'
_output_shapes
:���������d�
lstm_40/lstm_cell_40/Sigmoid_1Sigmoid#lstm_40/lstm_cell_40/split:output:1*
T0*'
_output_shapes
:���������d�
lstm_40/lstm_cell_40/mulMul"lstm_40/lstm_cell_40/Sigmoid_1:y:0lstm_40/zeros_1:output:0*
T0*'
_output_shapes
:���������dx
lstm_40/lstm_cell_40/ReluRelu#lstm_40/lstm_cell_40/split:output:2*
T0*'
_output_shapes
:���������d�
lstm_40/lstm_cell_40/mul_1Mul lstm_40/lstm_cell_40/Sigmoid:y:0'lstm_40/lstm_cell_40/Relu:activations:0*
T0*'
_output_shapes
:���������d�
lstm_40/lstm_cell_40/add_1AddV2lstm_40/lstm_cell_40/mul:z:0lstm_40/lstm_cell_40/mul_1:z:0*
T0*'
_output_shapes
:���������d�
lstm_40/lstm_cell_40/Sigmoid_2Sigmoid#lstm_40/lstm_cell_40/split:output:3*
T0*'
_output_shapes
:���������du
lstm_40/lstm_cell_40/Relu_1Relulstm_40/lstm_cell_40/add_1:z:0*
T0*'
_output_shapes
:���������d�
lstm_40/lstm_cell_40/mul_2Mul"lstm_40/lstm_cell_40/Sigmoid_2:y:0)lstm_40/lstm_cell_40/Relu_1:activations:0*
T0*'
_output_shapes
:���������dv
%lstm_40/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   f
$lstm_40/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_40/TensorArrayV2_1TensorListReserve.lstm_40/TensorArrayV2_1/element_shape:output:0-lstm_40/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���N
lstm_40/timeConst*
_output_shapes
: *
dtype0*
value	B : k
 lstm_40/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������\
lstm_40/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
lstm_40/whileWhile#lstm_40/while/loop_counter:output:0)lstm_40/while/maximum_iterations:output:0lstm_40/time:output:0 lstm_40/TensorArrayV2_1:handle:0lstm_40/zeros:output:0lstm_40/zeros_1:output:0 lstm_40/strided_slice_1:output:0?lstm_40/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_40_lstm_cell_40_matmul_readvariableop_resource5lstm_40_lstm_cell_40_matmul_1_readvariableop_resource4lstm_40_lstm_cell_40_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������d:���������d: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *&
bodyR
lstm_40_while_body_1378391*&
condR
lstm_40_while_cond_1378390*K
output_shapes:
8: : : : :���������d:���������d: : : : : *
parallel_iterations �
8lstm_40/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   �
*lstm_40/TensorArrayV2Stack/TensorListStackTensorListStacklstm_40/while:output:3Alstm_40/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������d*
element_dtype0*
num_elementsp
lstm_40/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������i
lstm_40/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: i
lstm_40/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
lstm_40/strided_slice_3StridedSlice3lstm_40/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_40/strided_slice_3/stack:output:0(lstm_40/strided_slice_3/stack_1:output:0(lstm_40/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������d*
shrink_axis_maskm
lstm_40/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
lstm_40/transpose_1	Transpose3lstm_40/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_40/transpose_1/perm:output:0*
T0*+
_output_shapes
:���������dc
lstm_40/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    �
dense_120/MatMul/ReadVariableOpReadVariableOp(dense_120_matmul_readvariableop_resource*
_output_shapes

:d2*
dtype0�
dense_120/MatMulMatMul lstm_40/strided_slice_3:output:0'dense_120/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2�
 dense_120/BiasAdd/ReadVariableOpReadVariableOp)dense_120_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0�
dense_120/BiasAddBiasAdddense_120/MatMul:product:0(dense_120/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2d
dense_120/ReluReludense_120/BiasAdd:output:0*
T0*'
_output_shapes
:���������2�
dense_121/MatMul/ReadVariableOpReadVariableOp(dense_121_matmul_readvariableop_resource*
_output_shapes

:22*
dtype0�
dense_121/MatMulMatMuldense_120/Relu:activations:0'dense_121/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2�
 dense_121/BiasAdd/ReadVariableOpReadVariableOp)dense_121_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0�
dense_121/BiasAddBiasAdddense_121/MatMul:product:0(dense_121/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2d
dense_121/ReluReludense_121/BiasAdd:output:0*
T0*'
_output_shapes
:���������2�
dense_122/MatMul/ReadVariableOpReadVariableOp(dense_122_matmul_readvariableop_resource*
_output_shapes

:2*
dtype0�
dense_122/MatMulMatMuldense_121/Relu:activations:0'dense_122/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
 dense_122/BiasAdd/ReadVariableOpReadVariableOp)dense_122_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_122/BiasAddBiasAdddense_122/MatMul:product:0(dense_122/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������i
IdentityIdentitydense_122/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp!^dense_120/BiasAdd/ReadVariableOp ^dense_120/MatMul/ReadVariableOp!^dense_121/BiasAdd/ReadVariableOp ^dense_121/MatMul/ReadVariableOp!^dense_122/BiasAdd/ReadVariableOp ^dense_122/MatMul/ReadVariableOp,^lstm_40/lstm_cell_40/BiasAdd/ReadVariableOp+^lstm_40/lstm_cell_40/MatMul/ReadVariableOp-^lstm_40/lstm_cell_40/MatMul_1/ReadVariableOp^lstm_40/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):���������: : : : : : : : : 2D
 dense_120/BiasAdd/ReadVariableOp dense_120/BiasAdd/ReadVariableOp2B
dense_120/MatMul/ReadVariableOpdense_120/MatMul/ReadVariableOp2D
 dense_121/BiasAdd/ReadVariableOp dense_121/BiasAdd/ReadVariableOp2B
dense_121/MatMul/ReadVariableOpdense_121/MatMul/ReadVariableOp2D
 dense_122/BiasAdd/ReadVariableOp dense_122/BiasAdd/ReadVariableOp2B
dense_122/MatMul/ReadVariableOpdense_122/MatMul/ReadVariableOp2Z
+lstm_40/lstm_cell_40/BiasAdd/ReadVariableOp+lstm_40/lstm_cell_40/BiasAdd/ReadVariableOp2X
*lstm_40/lstm_cell_40/MatMul/ReadVariableOp*lstm_40/lstm_cell_40/MatMul/ReadVariableOp2\
,lstm_40/lstm_cell_40/MatMul_1/ReadVariableOp,lstm_40/lstm_cell_40/MatMul_1/ReadVariableOp2
lstm_40/whilelstm_40/while:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
I__inference_lstm_cell_40_layer_call_and_return_conditional_losses_1379410

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
F__inference_dense_120_layer_call_and_return_conditional_losses_1379305

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
�$
�
while_body_1377600
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0/
while_lstm_cell_40_1377624_0:	�/
while_lstm_cell_40_1377626_0:	d�+
while_lstm_cell_40_1377628_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor-
while_lstm_cell_40_1377624:	�-
while_lstm_cell_40_1377626:	d�)
while_lstm_cell_40_1377628:	���*while/lstm_cell_40/StatefulPartitionedCall�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
*while/lstm_cell_40/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_40_1377624_0while_lstm_cell_40_1377626_0while_lstm_cell_40_1377628_0*
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
GPU 2J 8� *R
fMRK
I__inference_lstm_cell_40_layer_call_and_return_conditional_losses_1377540r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : �
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:03while/lstm_cell_40/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity3while/lstm_cell_40/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:���������d�
while/Identity_5Identity3while/lstm_cell_40/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:���������dy

while/NoOpNoOp+^while/lstm_cell_40/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0":
while_lstm_cell_40_1377624while_lstm_cell_40_1377624_0":
while_lstm_cell_40_1377626while_lstm_cell_40_1377626_0":
while_lstm_cell_40_1377628while_lstm_cell_40_1377628_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������d:���������d: : : : : 2X
*while/lstm_cell_40/StatefulPartitionedCall*while/lstm_cell_40/StatefulPartitionedCall: 
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
�
�
+__inference_dense_122_layer_call_fn_1379334

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
GPU 2J 8� *O
fJRH
F__inference_dense_122_layer_call_and_return_conditional_losses_1377882o
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
 
_user_specified_nameinputs
�

�
lstm_40_while_cond_1378555,
(lstm_40_while_lstm_40_while_loop_counter2
.lstm_40_while_lstm_40_while_maximum_iterations
lstm_40_while_placeholder
lstm_40_while_placeholder_1
lstm_40_while_placeholder_2
lstm_40_while_placeholder_3.
*lstm_40_while_less_lstm_40_strided_slice_1E
Alstm_40_while_lstm_40_while_cond_1378555___redundant_placeholder0E
Alstm_40_while_lstm_40_while_cond_1378555___redundant_placeholder1E
Alstm_40_while_lstm_40_while_cond_1378555___redundant_placeholder2E
Alstm_40_while_lstm_40_while_cond_1378555___redundant_placeholder3
lstm_40_while_identity
�
lstm_40/while/LessLesslstm_40_while_placeholder*lstm_40_while_less_lstm_40_strided_slice_1*
T0*
_output_shapes
: [
lstm_40/while/IdentityIdentitylstm_40/while/Less:z:0*
T0
*
_output_shapes
: "9
lstm_40_while_identitylstm_40/while/Identity:output:0*(
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
�
D__inference_lstm_40_layer_call_and_return_conditional_losses_1377477

inputs'
lstm_cell_40_1377393:	�'
lstm_cell_40_1377395:	d�#
lstm_cell_40_1377397:	�
identity��$lstm_cell_40/StatefulPartitionedCall�while;
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
shrink_axis_mask�
$lstm_cell_40/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_40_1377393lstm_cell_40_1377395lstm_cell_40_1377397*
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
GPU 2J 8� *R
fMRK
I__inference_lstm_cell_40_layer_call_and_return_conditional_losses_1377392n
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_40_1377393lstm_cell_40_1377395lstm_cell_40_1377397*
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
bodyR
while_body_1377407*
condR
while_cond_1377406*K
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
NoOpNoOp%^lstm_cell_40/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 2L
$lstm_cell_40/StatefulPartitionedCall$lstm_cell_40/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs
�
�
+__inference_dense_121_layer_call_fn_1379314

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
GPU 2J 8� *O
fJRH
F__inference_dense_121_layer_call_and_return_conditional_losses_1377866o
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
�
�
while_cond_1378764
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1378764___redundant_placeholder05
1while_while_cond_1378764___redundant_placeholder15
1while_while_cond_1378764___redundant_placeholder25
1while_while_cond_1378764___redundant_placeholder3
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
/__inference_sequential_40_layer_call_fn_1377910
lstm_40_input
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
StatefulPartitionedCallStatefulPartitionedCalllstm_40_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
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
GPU 2J 8� *S
fNRL
J__inference_sequential_40_layer_call_and_return_conditional_losses_1377889o
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
_user_specified_namelstm_40_input
�
�
)__inference_lstm_40_layer_call_fn_1378672
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
GPU 2J 8� *M
fHRF
D__inference_lstm_40_layer_call_and_return_conditional_losses_1377477o
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
while_cond_1379199
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1379199___redundant_placeholder05
1while_while_cond_1379199___redundant_placeholder15
1while_while_cond_1379199___redundant_placeholder25
1while_while_cond_1379199___redundant_placeholder3
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
while_body_1379055
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_40_matmul_readvariableop_resource_0:	�H
5while_lstm_cell_40_matmul_1_readvariableop_resource_0:	d�C
4while_lstm_cell_40_biasadd_readvariableop_resource_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_40_matmul_readvariableop_resource:	�F
3while_lstm_cell_40_matmul_1_readvariableop_resource:	d�A
2while_lstm_cell_40_biasadd_readvariableop_resource:	���)while/lstm_cell_40/BiasAdd/ReadVariableOp�(while/lstm_cell_40/MatMul/ReadVariableOp�*while/lstm_cell_40/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
(while/lstm_cell_40/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_40_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
while/lstm_cell_40/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_40/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
*while/lstm_cell_40/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_40_matmul_1_readvariableop_resource_0*
_output_shapes
:	d�*
dtype0�
while/lstm_cell_40/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_40/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
while/lstm_cell_40/addAddV2#while/lstm_cell_40/MatMul:product:0%while/lstm_cell_40/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
)while/lstm_cell_40/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_40_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype0�
while/lstm_cell_40/BiasAddBiasAddwhile/lstm_cell_40/add:z:01while/lstm_cell_40/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������d
"while/lstm_cell_40/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/lstm_cell_40/splitSplit+while/lstm_cell_40/split/split_dim:output:0#while/lstm_cell_40/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������d:���������d:���������d:���������d*
	num_splitz
while/lstm_cell_40/SigmoidSigmoid!while/lstm_cell_40/split:output:0*
T0*'
_output_shapes
:���������d|
while/lstm_cell_40/Sigmoid_1Sigmoid!while/lstm_cell_40/split:output:1*
T0*'
_output_shapes
:���������d�
while/lstm_cell_40/mulMul while/lstm_cell_40/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������dt
while/lstm_cell_40/ReluRelu!while/lstm_cell_40/split:output:2*
T0*'
_output_shapes
:���������d�
while/lstm_cell_40/mul_1Mulwhile/lstm_cell_40/Sigmoid:y:0%while/lstm_cell_40/Relu:activations:0*
T0*'
_output_shapes
:���������d�
while/lstm_cell_40/add_1AddV2while/lstm_cell_40/mul:z:0while/lstm_cell_40/mul_1:z:0*
T0*'
_output_shapes
:���������d|
while/lstm_cell_40/Sigmoid_2Sigmoid!while/lstm_cell_40/split:output:3*
T0*'
_output_shapes
:���������dq
while/lstm_cell_40/Relu_1Reluwhile/lstm_cell_40/add_1:z:0*
T0*'
_output_shapes
:���������d�
while/lstm_cell_40/mul_2Mul while/lstm_cell_40/Sigmoid_2:y:0'while/lstm_cell_40/Relu_1:activations:0*
T0*'
_output_shapes
:���������dr
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : �
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell_40/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_40/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:���������dy
while/Identity_5Identitywhile/lstm_cell_40/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:���������d�

while/NoOpNoOp*^while/lstm_cell_40/BiasAdd/ReadVariableOp)^while/lstm_cell_40/MatMul/ReadVariableOp+^while/lstm_cell_40/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_40_biasadd_readvariableop_resource4while_lstm_cell_40_biasadd_readvariableop_resource_0"l
3while_lstm_cell_40_matmul_1_readvariableop_resource5while_lstm_cell_40_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_40_matmul_readvariableop_resource3while_lstm_cell_40_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������d:���������d: : : : : 2V
)while/lstm_cell_40/BiasAdd/ReadVariableOp)while/lstm_cell_40/BiasAdd/ReadVariableOp2T
(while/lstm_cell_40/MatMul/ReadVariableOp(while/lstm_cell_40/MatMul/ReadVariableOp2X
*while/lstm_cell_40/MatMul_1/ReadVariableOp*while/lstm_cell_40/MatMul_1/ReadVariableOp: 
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
while_cond_1378909
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1378909___redundant_placeholder05
1while_while_cond_1378909___redundant_placeholder15
1while_while_cond_1378909___redundant_placeholder25
1while_while_cond_1378909___redundant_placeholder3
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
�K
�
D__inference_lstm_40_layer_call_and_return_conditional_losses_1378850
inputs_0>
+lstm_cell_40_matmul_readvariableop_resource:	�@
-lstm_cell_40_matmul_1_readvariableop_resource:	d�;
,lstm_cell_40_biasadd_readvariableop_resource:	�
identity��#lstm_cell_40/BiasAdd/ReadVariableOp�"lstm_cell_40/MatMul/ReadVariableOp�$lstm_cell_40/MatMul_1/ReadVariableOp�while=
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
"lstm_cell_40/MatMul/ReadVariableOpReadVariableOp+lstm_cell_40_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
lstm_cell_40/MatMulMatMulstrided_slice_2:output:0*lstm_cell_40/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
$lstm_cell_40/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_40_matmul_1_readvariableop_resource*
_output_shapes
:	d�*
dtype0�
lstm_cell_40/MatMul_1MatMulzeros:output:0,lstm_cell_40/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
lstm_cell_40/addAddV2lstm_cell_40/MatMul:product:0lstm_cell_40/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
#lstm_cell_40/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_40_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
lstm_cell_40/BiasAddBiasAddlstm_cell_40/add:z:0+lstm_cell_40/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������^
lstm_cell_40/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_cell_40/splitSplit%lstm_cell_40/split/split_dim:output:0lstm_cell_40/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������d:���������d:���������d:���������d*
	num_splitn
lstm_cell_40/SigmoidSigmoidlstm_cell_40/split:output:0*
T0*'
_output_shapes
:���������dp
lstm_cell_40/Sigmoid_1Sigmoidlstm_cell_40/split:output:1*
T0*'
_output_shapes
:���������dw
lstm_cell_40/mulMullstm_cell_40/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������dh
lstm_cell_40/ReluRelulstm_cell_40/split:output:2*
T0*'
_output_shapes
:���������d�
lstm_cell_40/mul_1Mullstm_cell_40/Sigmoid:y:0lstm_cell_40/Relu:activations:0*
T0*'
_output_shapes
:���������d{
lstm_cell_40/add_1AddV2lstm_cell_40/mul:z:0lstm_cell_40/mul_1:z:0*
T0*'
_output_shapes
:���������dp
lstm_cell_40/Sigmoid_2Sigmoidlstm_cell_40/split:output:3*
T0*'
_output_shapes
:���������de
lstm_cell_40/Relu_1Relulstm_cell_40/add_1:z:0*
T0*'
_output_shapes
:���������d�
lstm_cell_40/mul_2Mullstm_cell_40/Sigmoid_2:y:0!lstm_cell_40/Relu_1:activations:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_40_matmul_readvariableop_resource-lstm_cell_40_matmul_1_readvariableop_resource,lstm_cell_40_biasadd_readvariableop_resource*
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
bodyR
while_body_1378765*
condR
while_cond_1378764*K
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
NoOpNoOp$^lstm_cell_40/BiasAdd/ReadVariableOp#^lstm_cell_40/MatMul/ReadVariableOp%^lstm_cell_40/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 2J
#lstm_cell_40/BiasAdd/ReadVariableOp#lstm_cell_40/BiasAdd/ReadVariableOp2H
"lstm_cell_40/MatMul/ReadVariableOp"lstm_cell_40/MatMul/ReadVariableOp2L
$lstm_cell_40/MatMul_1/ReadVariableOp$lstm_cell_40/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :������������������
"
_user_specified_name
inputs/0
�	
�
%__inference_signature_wrapper_1378285
lstm_40_input
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
StatefulPartitionedCallStatefulPartitionedCalllstm_40_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
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
GPU 2J 8� *+
f&R$
"__inference__wrapped_model_1377325o
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
_user_specified_namelstm_40_input
�9
�
while_body_1378013
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_40_matmul_readvariableop_resource_0:	�H
5while_lstm_cell_40_matmul_1_readvariableop_resource_0:	d�C
4while_lstm_cell_40_biasadd_readvariableop_resource_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_40_matmul_readvariableop_resource:	�F
3while_lstm_cell_40_matmul_1_readvariableop_resource:	d�A
2while_lstm_cell_40_biasadd_readvariableop_resource:	���)while/lstm_cell_40/BiasAdd/ReadVariableOp�(while/lstm_cell_40/MatMul/ReadVariableOp�*while/lstm_cell_40/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
(while/lstm_cell_40/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_40_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
while/lstm_cell_40/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_40/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
*while/lstm_cell_40/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_40_matmul_1_readvariableop_resource_0*
_output_shapes
:	d�*
dtype0�
while/lstm_cell_40/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_40/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
while/lstm_cell_40/addAddV2#while/lstm_cell_40/MatMul:product:0%while/lstm_cell_40/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
)while/lstm_cell_40/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_40_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype0�
while/lstm_cell_40/BiasAddBiasAddwhile/lstm_cell_40/add:z:01while/lstm_cell_40/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������d
"while/lstm_cell_40/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/lstm_cell_40/splitSplit+while/lstm_cell_40/split/split_dim:output:0#while/lstm_cell_40/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������d:���������d:���������d:���������d*
	num_splitz
while/lstm_cell_40/SigmoidSigmoid!while/lstm_cell_40/split:output:0*
T0*'
_output_shapes
:���������d|
while/lstm_cell_40/Sigmoid_1Sigmoid!while/lstm_cell_40/split:output:1*
T0*'
_output_shapes
:���������d�
while/lstm_cell_40/mulMul while/lstm_cell_40/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������dt
while/lstm_cell_40/ReluRelu!while/lstm_cell_40/split:output:2*
T0*'
_output_shapes
:���������d�
while/lstm_cell_40/mul_1Mulwhile/lstm_cell_40/Sigmoid:y:0%while/lstm_cell_40/Relu:activations:0*
T0*'
_output_shapes
:���������d�
while/lstm_cell_40/add_1AddV2while/lstm_cell_40/mul:z:0while/lstm_cell_40/mul_1:z:0*
T0*'
_output_shapes
:���������d|
while/lstm_cell_40/Sigmoid_2Sigmoid!while/lstm_cell_40/split:output:3*
T0*'
_output_shapes
:���������dq
while/lstm_cell_40/Relu_1Reluwhile/lstm_cell_40/add_1:z:0*
T0*'
_output_shapes
:���������d�
while/lstm_cell_40/mul_2Mul while/lstm_cell_40/Sigmoid_2:y:0'while/lstm_cell_40/Relu_1:activations:0*
T0*'
_output_shapes
:���������dr
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : �
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell_40/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_40/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:���������dy
while/Identity_5Identitywhile/lstm_cell_40/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:���������d�

while/NoOpNoOp*^while/lstm_cell_40/BiasAdd/ReadVariableOp)^while/lstm_cell_40/MatMul/ReadVariableOp+^while/lstm_cell_40/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_40_biasadd_readvariableop_resource4while_lstm_cell_40_biasadd_readvariableop_resource_0"l
3while_lstm_cell_40_matmul_1_readvariableop_resource5while_lstm_cell_40_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_40_matmul_readvariableop_resource3while_lstm_cell_40_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������d:���������d: : : : : 2V
)while/lstm_cell_40/BiasAdd/ReadVariableOp)while/lstm_cell_40/BiasAdd/ReadVariableOp2T
(while/lstm_cell_40/MatMul/ReadVariableOp(while/lstm_cell_40/MatMul/ReadVariableOp2X
*while/lstm_cell_40/MatMul_1/ReadVariableOp*while/lstm_cell_40/MatMul_1/ReadVariableOp: 
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
�
�
J__inference_sequential_40_layer_call_and_return_conditional_losses_1378158

inputs"
lstm_40_1378135:	�"
lstm_40_1378137:	d�
lstm_40_1378139:	�#
dense_120_1378142:d2
dense_120_1378144:2#
dense_121_1378147:22
dense_121_1378149:2#
dense_122_1378152:2
dense_122_1378154:
identity��!dense_120/StatefulPartitionedCall�!dense_121/StatefulPartitionedCall�!dense_122/StatefulPartitionedCall�lstm_40/StatefulPartitionedCall�
lstm_40/StatefulPartitionedCallStatefulPartitionedCallinputslstm_40_1378135lstm_40_1378137lstm_40_1378139*
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
GPU 2J 8� *M
fHRF
D__inference_lstm_40_layer_call_and_return_conditional_losses_1378098�
!dense_120/StatefulPartitionedCallStatefulPartitionedCall(lstm_40/StatefulPartitionedCall:output:0dense_120_1378142dense_120_1378144*
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
GPU 2J 8� *O
fJRH
F__inference_dense_120_layer_call_and_return_conditional_losses_1377849�
!dense_121/StatefulPartitionedCallStatefulPartitionedCall*dense_120/StatefulPartitionedCall:output:0dense_121_1378147dense_121_1378149*
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
GPU 2J 8� *O
fJRH
F__inference_dense_121_layer_call_and_return_conditional_losses_1377866�
!dense_122/StatefulPartitionedCallStatefulPartitionedCall*dense_121/StatefulPartitionedCall:output:0dense_122_1378152dense_122_1378154*
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
GPU 2J 8� *O
fJRH
F__inference_dense_122_layer_call_and_return_conditional_losses_1377882y
IdentityIdentity*dense_122/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp"^dense_120/StatefulPartitionedCall"^dense_121/StatefulPartitionedCall"^dense_122/StatefulPartitionedCall ^lstm_40/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):���������: : : : : : : : : 2F
!dense_120/StatefulPartitionedCall!dense_120/StatefulPartitionedCall2F
!dense_121/StatefulPartitionedCall!dense_121/StatefulPartitionedCall2F
!dense_122/StatefulPartitionedCall!dense_122/StatefulPartitionedCall2B
lstm_40/StatefulPartitionedCalllstm_40/StatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�9
�
while_body_1379200
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_40_matmul_readvariableop_resource_0:	�H
5while_lstm_cell_40_matmul_1_readvariableop_resource_0:	d�C
4while_lstm_cell_40_biasadd_readvariableop_resource_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_40_matmul_readvariableop_resource:	�F
3while_lstm_cell_40_matmul_1_readvariableop_resource:	d�A
2while_lstm_cell_40_biasadd_readvariableop_resource:	���)while/lstm_cell_40/BiasAdd/ReadVariableOp�(while/lstm_cell_40/MatMul/ReadVariableOp�*while/lstm_cell_40/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
(while/lstm_cell_40/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_40_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
while/lstm_cell_40/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_40/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
*while/lstm_cell_40/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_40_matmul_1_readvariableop_resource_0*
_output_shapes
:	d�*
dtype0�
while/lstm_cell_40/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_40/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
while/lstm_cell_40/addAddV2#while/lstm_cell_40/MatMul:product:0%while/lstm_cell_40/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
)while/lstm_cell_40/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_40_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype0�
while/lstm_cell_40/BiasAddBiasAddwhile/lstm_cell_40/add:z:01while/lstm_cell_40/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������d
"while/lstm_cell_40/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/lstm_cell_40/splitSplit+while/lstm_cell_40/split/split_dim:output:0#while/lstm_cell_40/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������d:���������d:���������d:���������d*
	num_splitz
while/lstm_cell_40/SigmoidSigmoid!while/lstm_cell_40/split:output:0*
T0*'
_output_shapes
:���������d|
while/lstm_cell_40/Sigmoid_1Sigmoid!while/lstm_cell_40/split:output:1*
T0*'
_output_shapes
:���������d�
while/lstm_cell_40/mulMul while/lstm_cell_40/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������dt
while/lstm_cell_40/ReluRelu!while/lstm_cell_40/split:output:2*
T0*'
_output_shapes
:���������d�
while/lstm_cell_40/mul_1Mulwhile/lstm_cell_40/Sigmoid:y:0%while/lstm_cell_40/Relu:activations:0*
T0*'
_output_shapes
:���������d�
while/lstm_cell_40/add_1AddV2while/lstm_cell_40/mul:z:0while/lstm_cell_40/mul_1:z:0*
T0*'
_output_shapes
:���������d|
while/lstm_cell_40/Sigmoid_2Sigmoid!while/lstm_cell_40/split:output:3*
T0*'
_output_shapes
:���������dq
while/lstm_cell_40/Relu_1Reluwhile/lstm_cell_40/add_1:z:0*
T0*'
_output_shapes
:���������d�
while/lstm_cell_40/mul_2Mul while/lstm_cell_40/Sigmoid_2:y:0'while/lstm_cell_40/Relu_1:activations:0*
T0*'
_output_shapes
:���������dr
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : �
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell_40/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_40/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:���������dy
while/Identity_5Identitywhile/lstm_cell_40/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:���������d�

while/NoOpNoOp*^while/lstm_cell_40/BiasAdd/ReadVariableOp)^while/lstm_cell_40/MatMul/ReadVariableOp+^while/lstm_cell_40/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_40_biasadd_readvariableop_resource4while_lstm_cell_40_biasadd_readvariableop_resource_0"l
3while_lstm_cell_40_matmul_1_readvariableop_resource5while_lstm_cell_40_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_40_matmul_readvariableop_resource3while_lstm_cell_40_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������d:���������d: : : : : 2V
)while/lstm_cell_40/BiasAdd/ReadVariableOp)while/lstm_cell_40/BiasAdd/ReadVariableOp2T
(while/lstm_cell_40/MatMul/ReadVariableOp(while/lstm_cell_40/MatMul/ReadVariableOp2X
*while/lstm_cell_40/MatMul_1/ReadVariableOp*while/lstm_cell_40/MatMul_1/ReadVariableOp: 
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
�L
�
 __inference__traced_save_1379573
file_prefix/
+savev2_dense_120_kernel_read_readvariableop-
)savev2_dense_120_bias_read_readvariableop/
+savev2_dense_121_kernel_read_readvariableop-
)savev2_dense_121_bias_read_readvariableop/
+savev2_dense_122_kernel_read_readvariableop-
)savev2_dense_122_bias_read_readvariableop:
6savev2_lstm_40_lstm_cell_40_kernel_read_readvariableopD
@savev2_lstm_40_lstm_cell_40_recurrent_kernel_read_readvariableop8
4savev2_lstm_40_lstm_cell_40_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop6
2savev2_adam_dense_120_kernel_m_read_readvariableop4
0savev2_adam_dense_120_bias_m_read_readvariableop6
2savev2_adam_dense_121_kernel_m_read_readvariableop4
0savev2_adam_dense_121_bias_m_read_readvariableop6
2savev2_adam_dense_122_kernel_m_read_readvariableop4
0savev2_adam_dense_122_bias_m_read_readvariableopA
=savev2_adam_lstm_40_lstm_cell_40_kernel_m_read_readvariableopK
Gsavev2_adam_lstm_40_lstm_cell_40_recurrent_kernel_m_read_readvariableop?
;savev2_adam_lstm_40_lstm_cell_40_bias_m_read_readvariableop6
2savev2_adam_dense_120_kernel_v_read_readvariableop4
0savev2_adam_dense_120_bias_v_read_readvariableop6
2savev2_adam_dense_121_kernel_v_read_readvariableop4
0savev2_adam_dense_121_bias_v_read_readvariableop6
2savev2_adam_dense_122_kernel_v_read_readvariableop4
0savev2_adam_dense_122_bias_v_read_readvariableopA
=savev2_adam_lstm_40_lstm_cell_40_kernel_v_read_readvariableopK
Gsavev2_adam_lstm_40_lstm_cell_40_recurrent_kernel_v_read_readvariableop?
;savev2_adam_lstm_40_lstm_cell_40_bias_v_read_readvariableop
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
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_dense_120_kernel_read_readvariableop)savev2_dense_120_bias_read_readvariableop+savev2_dense_121_kernel_read_readvariableop)savev2_dense_121_bias_read_readvariableop+savev2_dense_122_kernel_read_readvariableop)savev2_dense_122_bias_read_readvariableop6savev2_lstm_40_lstm_cell_40_kernel_read_readvariableop@savev2_lstm_40_lstm_cell_40_recurrent_kernel_read_readvariableop4savev2_lstm_40_lstm_cell_40_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop2savev2_adam_dense_120_kernel_m_read_readvariableop0savev2_adam_dense_120_bias_m_read_readvariableop2savev2_adam_dense_121_kernel_m_read_readvariableop0savev2_adam_dense_121_bias_m_read_readvariableop2savev2_adam_dense_122_kernel_m_read_readvariableop0savev2_adam_dense_122_bias_m_read_readvariableop=savev2_adam_lstm_40_lstm_cell_40_kernel_m_read_readvariableopGsavev2_adam_lstm_40_lstm_cell_40_recurrent_kernel_m_read_readvariableop;savev2_adam_lstm_40_lstm_cell_40_bias_m_read_readvariableop2savev2_adam_dense_120_kernel_v_read_readvariableop0savev2_adam_dense_120_bias_v_read_readvariableop2savev2_adam_dense_121_kernel_v_read_readvariableop0savev2_adam_dense_121_bias_v_read_readvariableop2savev2_adam_dense_122_kernel_v_read_readvariableop0savev2_adam_dense_122_bias_v_read_readvariableop=savev2_adam_lstm_40_lstm_cell_40_kernel_v_read_readvariableopGsavev2_adam_lstm_40_lstm_cell_40_recurrent_kernel_v_read_readvariableop;savev2_adam_lstm_40_lstm_cell_40_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
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
�
�
(sequential_40_lstm_40_while_cond_1377219H
Dsequential_40_lstm_40_while_sequential_40_lstm_40_while_loop_counterN
Jsequential_40_lstm_40_while_sequential_40_lstm_40_while_maximum_iterations+
'sequential_40_lstm_40_while_placeholder-
)sequential_40_lstm_40_while_placeholder_1-
)sequential_40_lstm_40_while_placeholder_2-
)sequential_40_lstm_40_while_placeholder_3J
Fsequential_40_lstm_40_while_less_sequential_40_lstm_40_strided_slice_1a
]sequential_40_lstm_40_while_sequential_40_lstm_40_while_cond_1377219___redundant_placeholder0a
]sequential_40_lstm_40_while_sequential_40_lstm_40_while_cond_1377219___redundant_placeholder1a
]sequential_40_lstm_40_while_sequential_40_lstm_40_while_cond_1377219___redundant_placeholder2a
]sequential_40_lstm_40_while_sequential_40_lstm_40_while_cond_1377219___redundant_placeholder3(
$sequential_40_lstm_40_while_identity
�
 sequential_40/lstm_40/while/LessLess'sequential_40_lstm_40_while_placeholderFsequential_40_lstm_40_while_less_sequential_40_lstm_40_strided_slice_1*
T0*
_output_shapes
: w
$sequential_40/lstm_40/while/IdentityIdentity$sequential_40/lstm_40/while/Less:z:0*
T0
*
_output_shapes
: "U
$sequential_40_lstm_40_while_identity-sequential_40/lstm_40/while/Identity:output:0*(
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

�
F__inference_dense_121_layer_call_and_return_conditional_losses_1379325

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
�K
�
D__inference_lstm_40_layer_call_and_return_conditional_losses_1379140

inputs>
+lstm_cell_40_matmul_readvariableop_resource:	�@
-lstm_cell_40_matmul_1_readvariableop_resource:	d�;
,lstm_cell_40_biasadd_readvariableop_resource:	�
identity��#lstm_cell_40/BiasAdd/ReadVariableOp�"lstm_cell_40/MatMul/ReadVariableOp�$lstm_cell_40/MatMul_1/ReadVariableOp�while;
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
"lstm_cell_40/MatMul/ReadVariableOpReadVariableOp+lstm_cell_40_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
lstm_cell_40/MatMulMatMulstrided_slice_2:output:0*lstm_cell_40/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
$lstm_cell_40/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_40_matmul_1_readvariableop_resource*
_output_shapes
:	d�*
dtype0�
lstm_cell_40/MatMul_1MatMulzeros:output:0,lstm_cell_40/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
lstm_cell_40/addAddV2lstm_cell_40/MatMul:product:0lstm_cell_40/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
#lstm_cell_40/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_40_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
lstm_cell_40/BiasAddBiasAddlstm_cell_40/add:z:0+lstm_cell_40/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������^
lstm_cell_40/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_cell_40/splitSplit%lstm_cell_40/split/split_dim:output:0lstm_cell_40/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������d:���������d:���������d:���������d*
	num_splitn
lstm_cell_40/SigmoidSigmoidlstm_cell_40/split:output:0*
T0*'
_output_shapes
:���������dp
lstm_cell_40/Sigmoid_1Sigmoidlstm_cell_40/split:output:1*
T0*'
_output_shapes
:���������dw
lstm_cell_40/mulMullstm_cell_40/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������dh
lstm_cell_40/ReluRelulstm_cell_40/split:output:2*
T0*'
_output_shapes
:���������d�
lstm_cell_40/mul_1Mullstm_cell_40/Sigmoid:y:0lstm_cell_40/Relu:activations:0*
T0*'
_output_shapes
:���������d{
lstm_cell_40/add_1AddV2lstm_cell_40/mul:z:0lstm_cell_40/mul_1:z:0*
T0*'
_output_shapes
:���������dp
lstm_cell_40/Sigmoid_2Sigmoidlstm_cell_40/split:output:3*
T0*'
_output_shapes
:���������de
lstm_cell_40/Relu_1Relulstm_cell_40/add_1:z:0*
T0*'
_output_shapes
:���������d�
lstm_cell_40/mul_2Mullstm_cell_40/Sigmoid_2:y:0!lstm_cell_40/Relu_1:activations:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_40_matmul_readvariableop_resource-lstm_cell_40_matmul_1_readvariableop_resource,lstm_cell_40_biasadd_readvariableop_resource*
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
bodyR
while_body_1379055*
condR
while_cond_1379054*K
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
NoOpNoOp$^lstm_cell_40/BiasAdd/ReadVariableOp#^lstm_cell_40/MatMul/ReadVariableOp%^lstm_cell_40/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������: : : 2J
#lstm_cell_40/BiasAdd/ReadVariableOp#lstm_cell_40/BiasAdd/ReadVariableOp2H
"lstm_cell_40/MatMul/ReadVariableOp"lstm_cell_40/MatMul/ReadVariableOp2L
$lstm_cell_40/MatMul_1/ReadVariableOp$lstm_cell_40/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
while_cond_1379054
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1379054___redundant_placeholder05
1while_while_cond_1379054___redundant_placeholder15
1while_while_cond_1379054___redundant_placeholder25
1while_while_cond_1379054___redundant_placeholder3
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
while_body_1377745
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_40_matmul_readvariableop_resource_0:	�H
5while_lstm_cell_40_matmul_1_readvariableop_resource_0:	d�C
4while_lstm_cell_40_biasadd_readvariableop_resource_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_40_matmul_readvariableop_resource:	�F
3while_lstm_cell_40_matmul_1_readvariableop_resource:	d�A
2while_lstm_cell_40_biasadd_readvariableop_resource:	���)while/lstm_cell_40/BiasAdd/ReadVariableOp�(while/lstm_cell_40/MatMul/ReadVariableOp�*while/lstm_cell_40/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
(while/lstm_cell_40/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_40_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
while/lstm_cell_40/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_40/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
*while/lstm_cell_40/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_40_matmul_1_readvariableop_resource_0*
_output_shapes
:	d�*
dtype0�
while/lstm_cell_40/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_40/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
while/lstm_cell_40/addAddV2#while/lstm_cell_40/MatMul:product:0%while/lstm_cell_40/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
)while/lstm_cell_40/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_40_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype0�
while/lstm_cell_40/BiasAddBiasAddwhile/lstm_cell_40/add:z:01while/lstm_cell_40/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������d
"while/lstm_cell_40/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/lstm_cell_40/splitSplit+while/lstm_cell_40/split/split_dim:output:0#while/lstm_cell_40/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������d:���������d:���������d:���������d*
	num_splitz
while/lstm_cell_40/SigmoidSigmoid!while/lstm_cell_40/split:output:0*
T0*'
_output_shapes
:���������d|
while/lstm_cell_40/Sigmoid_1Sigmoid!while/lstm_cell_40/split:output:1*
T0*'
_output_shapes
:���������d�
while/lstm_cell_40/mulMul while/lstm_cell_40/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������dt
while/lstm_cell_40/ReluRelu!while/lstm_cell_40/split:output:2*
T0*'
_output_shapes
:���������d�
while/lstm_cell_40/mul_1Mulwhile/lstm_cell_40/Sigmoid:y:0%while/lstm_cell_40/Relu:activations:0*
T0*'
_output_shapes
:���������d�
while/lstm_cell_40/add_1AddV2while/lstm_cell_40/mul:z:0while/lstm_cell_40/mul_1:z:0*
T0*'
_output_shapes
:���������d|
while/lstm_cell_40/Sigmoid_2Sigmoid!while/lstm_cell_40/split:output:3*
T0*'
_output_shapes
:���������dq
while/lstm_cell_40/Relu_1Reluwhile/lstm_cell_40/add_1:z:0*
T0*'
_output_shapes
:���������d�
while/lstm_cell_40/mul_2Mul while/lstm_cell_40/Sigmoid_2:y:0'while/lstm_cell_40/Relu_1:activations:0*
T0*'
_output_shapes
:���������dr
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : �
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell_40/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_40/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:���������dy
while/Identity_5Identitywhile/lstm_cell_40/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:���������d�

while/NoOpNoOp*^while/lstm_cell_40/BiasAdd/ReadVariableOp)^while/lstm_cell_40/MatMul/ReadVariableOp+^while/lstm_cell_40/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_40_biasadd_readvariableop_resource4while_lstm_cell_40_biasadd_readvariableop_resource_0"l
3while_lstm_cell_40_matmul_1_readvariableop_resource5while_lstm_cell_40_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_40_matmul_readvariableop_resource3while_lstm_cell_40_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������d:���������d: : : : : 2V
)while/lstm_cell_40/BiasAdd/ReadVariableOp)while/lstm_cell_40/BiasAdd/ReadVariableOp2T
(while/lstm_cell_40/MatMul/ReadVariableOp(while/lstm_cell_40/MatMul/ReadVariableOp2X
*while/lstm_cell_40/MatMul_1/ReadVariableOp*while/lstm_cell_40/MatMul_1/ReadVariableOp: 
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
while_body_1378910
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_40_matmul_readvariableop_resource_0:	�H
5while_lstm_cell_40_matmul_1_readvariableop_resource_0:	d�C
4while_lstm_cell_40_biasadd_readvariableop_resource_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_40_matmul_readvariableop_resource:	�F
3while_lstm_cell_40_matmul_1_readvariableop_resource:	d�A
2while_lstm_cell_40_biasadd_readvariableop_resource:	���)while/lstm_cell_40/BiasAdd/ReadVariableOp�(while/lstm_cell_40/MatMul/ReadVariableOp�*while/lstm_cell_40/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
(while/lstm_cell_40/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_40_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
while/lstm_cell_40/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_40/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
*while/lstm_cell_40/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_40_matmul_1_readvariableop_resource_0*
_output_shapes
:	d�*
dtype0�
while/lstm_cell_40/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_40/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
while/lstm_cell_40/addAddV2#while/lstm_cell_40/MatMul:product:0%while/lstm_cell_40/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
)while/lstm_cell_40/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_40_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype0�
while/lstm_cell_40/BiasAddBiasAddwhile/lstm_cell_40/add:z:01while/lstm_cell_40/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������d
"while/lstm_cell_40/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/lstm_cell_40/splitSplit+while/lstm_cell_40/split/split_dim:output:0#while/lstm_cell_40/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������d:���������d:���������d:���������d*
	num_splitz
while/lstm_cell_40/SigmoidSigmoid!while/lstm_cell_40/split:output:0*
T0*'
_output_shapes
:���������d|
while/lstm_cell_40/Sigmoid_1Sigmoid!while/lstm_cell_40/split:output:1*
T0*'
_output_shapes
:���������d�
while/lstm_cell_40/mulMul while/lstm_cell_40/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������dt
while/lstm_cell_40/ReluRelu!while/lstm_cell_40/split:output:2*
T0*'
_output_shapes
:���������d�
while/lstm_cell_40/mul_1Mulwhile/lstm_cell_40/Sigmoid:y:0%while/lstm_cell_40/Relu:activations:0*
T0*'
_output_shapes
:���������d�
while/lstm_cell_40/add_1AddV2while/lstm_cell_40/mul:z:0while/lstm_cell_40/mul_1:z:0*
T0*'
_output_shapes
:���������d|
while/lstm_cell_40/Sigmoid_2Sigmoid!while/lstm_cell_40/split:output:3*
T0*'
_output_shapes
:���������dq
while/lstm_cell_40/Relu_1Reluwhile/lstm_cell_40/add_1:z:0*
T0*'
_output_shapes
:���������d�
while/lstm_cell_40/mul_2Mul while/lstm_cell_40/Sigmoid_2:y:0'while/lstm_cell_40/Relu_1:activations:0*
T0*'
_output_shapes
:���������dr
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : �
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell_40/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_40/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:���������dy
while/Identity_5Identitywhile/lstm_cell_40/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:���������d�

while/NoOpNoOp*^while/lstm_cell_40/BiasAdd/ReadVariableOp)^while/lstm_cell_40/MatMul/ReadVariableOp+^while/lstm_cell_40/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_40_biasadd_readvariableop_resource4while_lstm_cell_40_biasadd_readvariableop_resource_0"l
3while_lstm_cell_40_matmul_1_readvariableop_resource5while_lstm_cell_40_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_40_matmul_readvariableop_resource3while_lstm_cell_40_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������d:���������d: : : : : 2V
)while/lstm_cell_40/BiasAdd/ReadVariableOp)while/lstm_cell_40/BiasAdd/ReadVariableOp2T
(while/lstm_cell_40/MatMul/ReadVariableOp(while/lstm_cell_40/MatMul/ReadVariableOp2X
*while/lstm_cell_40/MatMul_1/ReadVariableOp*while/lstm_cell_40/MatMul_1/ReadVariableOp: 
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
D__inference_lstm_40_layer_call_and_return_conditional_losses_1378098

inputs>
+lstm_cell_40_matmul_readvariableop_resource:	�@
-lstm_cell_40_matmul_1_readvariableop_resource:	d�;
,lstm_cell_40_biasadd_readvariableop_resource:	�
identity��#lstm_cell_40/BiasAdd/ReadVariableOp�"lstm_cell_40/MatMul/ReadVariableOp�$lstm_cell_40/MatMul_1/ReadVariableOp�while;
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
"lstm_cell_40/MatMul/ReadVariableOpReadVariableOp+lstm_cell_40_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
lstm_cell_40/MatMulMatMulstrided_slice_2:output:0*lstm_cell_40/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
$lstm_cell_40/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_40_matmul_1_readvariableop_resource*
_output_shapes
:	d�*
dtype0�
lstm_cell_40/MatMul_1MatMulzeros:output:0,lstm_cell_40/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
lstm_cell_40/addAddV2lstm_cell_40/MatMul:product:0lstm_cell_40/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
#lstm_cell_40/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_40_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
lstm_cell_40/BiasAddBiasAddlstm_cell_40/add:z:0+lstm_cell_40/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������^
lstm_cell_40/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_cell_40/splitSplit%lstm_cell_40/split/split_dim:output:0lstm_cell_40/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������d:���������d:���������d:���������d*
	num_splitn
lstm_cell_40/SigmoidSigmoidlstm_cell_40/split:output:0*
T0*'
_output_shapes
:���������dp
lstm_cell_40/Sigmoid_1Sigmoidlstm_cell_40/split:output:1*
T0*'
_output_shapes
:���������dw
lstm_cell_40/mulMullstm_cell_40/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������dh
lstm_cell_40/ReluRelulstm_cell_40/split:output:2*
T0*'
_output_shapes
:���������d�
lstm_cell_40/mul_1Mullstm_cell_40/Sigmoid:y:0lstm_cell_40/Relu:activations:0*
T0*'
_output_shapes
:���������d{
lstm_cell_40/add_1AddV2lstm_cell_40/mul:z:0lstm_cell_40/mul_1:z:0*
T0*'
_output_shapes
:���������dp
lstm_cell_40/Sigmoid_2Sigmoidlstm_cell_40/split:output:3*
T0*'
_output_shapes
:���������de
lstm_cell_40/Relu_1Relulstm_cell_40/add_1:z:0*
T0*'
_output_shapes
:���������d�
lstm_cell_40/mul_2Mullstm_cell_40/Sigmoid_2:y:0!lstm_cell_40/Relu_1:activations:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_40_matmul_readvariableop_resource-lstm_cell_40_matmul_1_readvariableop_resource,lstm_cell_40_biasadd_readvariableop_resource*
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
bodyR
while_body_1378013*
condR
while_cond_1378012*K
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
NoOpNoOp$^lstm_cell_40/BiasAdd/ReadVariableOp#^lstm_cell_40/MatMul/ReadVariableOp%^lstm_cell_40/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������: : : 2J
#lstm_cell_40/BiasAdd/ReadVariableOp#lstm_cell_40/BiasAdd/ReadVariableOp2H
"lstm_cell_40/MatMul/ReadVariableOp"lstm_cell_40/MatMul/ReadVariableOp2L
$lstm_cell_40/MatMul_1/ReadVariableOp$lstm_cell_40/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
+__inference_dense_120_layer_call_fn_1379294

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
GPU 2J 8� *O
fJRH
F__inference_dense_120_layer_call_and_return_conditional_losses_1377849o
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
�
�
J__inference_sequential_40_layer_call_and_return_conditional_losses_1377889

inputs"
lstm_40_1377831:	�"
lstm_40_1377833:	d�
lstm_40_1377835:	�#
dense_120_1377850:d2
dense_120_1377852:2#
dense_121_1377867:22
dense_121_1377869:2#
dense_122_1377883:2
dense_122_1377885:
identity��!dense_120/StatefulPartitionedCall�!dense_121/StatefulPartitionedCall�!dense_122/StatefulPartitionedCall�lstm_40/StatefulPartitionedCall�
lstm_40/StatefulPartitionedCallStatefulPartitionedCallinputslstm_40_1377831lstm_40_1377833lstm_40_1377835*
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
GPU 2J 8� *M
fHRF
D__inference_lstm_40_layer_call_and_return_conditional_losses_1377830�
!dense_120/StatefulPartitionedCallStatefulPartitionedCall(lstm_40/StatefulPartitionedCall:output:0dense_120_1377850dense_120_1377852*
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
GPU 2J 8� *O
fJRH
F__inference_dense_120_layer_call_and_return_conditional_losses_1377849�
!dense_121/StatefulPartitionedCallStatefulPartitionedCall*dense_120/StatefulPartitionedCall:output:0dense_121_1377867dense_121_1377869*
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
GPU 2J 8� *O
fJRH
F__inference_dense_121_layer_call_and_return_conditional_losses_1377866�
!dense_122/StatefulPartitionedCallStatefulPartitionedCall*dense_121/StatefulPartitionedCall:output:0dense_122_1377883dense_122_1377885*
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
GPU 2J 8� *O
fJRH
F__inference_dense_122_layer_call_and_return_conditional_losses_1377882y
IdentityIdentity*dense_122/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp"^dense_120/StatefulPartitionedCall"^dense_121/StatefulPartitionedCall"^dense_122/StatefulPartitionedCall ^lstm_40/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):���������: : : : : : : : : 2F
!dense_120/StatefulPartitionedCall!dense_120/StatefulPartitionedCall2F
!dense_121/StatefulPartitionedCall!dense_121/StatefulPartitionedCall2F
!dense_122/StatefulPartitionedCall!dense_122/StatefulPartitionedCall2B
lstm_40/StatefulPartitionedCalllstm_40/StatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�9
�
D__inference_lstm_40_layer_call_and_return_conditional_losses_1377670

inputs'
lstm_cell_40_1377586:	�'
lstm_cell_40_1377588:	d�#
lstm_cell_40_1377590:	�
identity��$lstm_cell_40/StatefulPartitionedCall�while;
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
shrink_axis_mask�
$lstm_cell_40/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_40_1377586lstm_cell_40_1377588lstm_cell_40_1377590*
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
GPU 2J 8� *R
fMRK
I__inference_lstm_cell_40_layer_call_and_return_conditional_losses_1377540n
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_40_1377586lstm_cell_40_1377588lstm_cell_40_1377590*
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
bodyR
while_body_1377600*
condR
while_cond_1377599*K
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
NoOpNoOp%^lstm_cell_40/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 2L
$lstm_cell_40/StatefulPartitionedCall$lstm_cell_40/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs
�B
�

lstm_40_while_body_1378556,
(lstm_40_while_lstm_40_while_loop_counter2
.lstm_40_while_lstm_40_while_maximum_iterations
lstm_40_while_placeholder
lstm_40_while_placeholder_1
lstm_40_while_placeholder_2
lstm_40_while_placeholder_3+
'lstm_40_while_lstm_40_strided_slice_1_0g
clstm_40_while_tensorarrayv2read_tensorlistgetitem_lstm_40_tensorarrayunstack_tensorlistfromtensor_0N
;lstm_40_while_lstm_cell_40_matmul_readvariableop_resource_0:	�P
=lstm_40_while_lstm_cell_40_matmul_1_readvariableop_resource_0:	d�K
<lstm_40_while_lstm_cell_40_biasadd_readvariableop_resource_0:	�
lstm_40_while_identity
lstm_40_while_identity_1
lstm_40_while_identity_2
lstm_40_while_identity_3
lstm_40_while_identity_4
lstm_40_while_identity_5)
%lstm_40_while_lstm_40_strided_slice_1e
alstm_40_while_tensorarrayv2read_tensorlistgetitem_lstm_40_tensorarrayunstack_tensorlistfromtensorL
9lstm_40_while_lstm_cell_40_matmul_readvariableop_resource:	�N
;lstm_40_while_lstm_cell_40_matmul_1_readvariableop_resource:	d�I
:lstm_40_while_lstm_cell_40_biasadd_readvariableop_resource:	���1lstm_40/while/lstm_cell_40/BiasAdd/ReadVariableOp�0lstm_40/while/lstm_cell_40/MatMul/ReadVariableOp�2lstm_40/while/lstm_cell_40/MatMul_1/ReadVariableOp�
?lstm_40/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
1lstm_40/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_40_while_tensorarrayv2read_tensorlistgetitem_lstm_40_tensorarrayunstack_tensorlistfromtensor_0lstm_40_while_placeholderHlstm_40/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
0lstm_40/while/lstm_cell_40/MatMul/ReadVariableOpReadVariableOp;lstm_40_while_lstm_cell_40_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
!lstm_40/while/lstm_cell_40/MatMulMatMul8lstm_40/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_40/while/lstm_cell_40/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
2lstm_40/while/lstm_cell_40/MatMul_1/ReadVariableOpReadVariableOp=lstm_40_while_lstm_cell_40_matmul_1_readvariableop_resource_0*
_output_shapes
:	d�*
dtype0�
#lstm_40/while/lstm_cell_40/MatMul_1MatMullstm_40_while_placeholder_2:lstm_40/while/lstm_cell_40/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
lstm_40/while/lstm_cell_40/addAddV2+lstm_40/while/lstm_cell_40/MatMul:product:0-lstm_40/while/lstm_cell_40/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
1lstm_40/while/lstm_cell_40/BiasAdd/ReadVariableOpReadVariableOp<lstm_40_while_lstm_cell_40_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype0�
"lstm_40/while/lstm_cell_40/BiasAddBiasAdd"lstm_40/while/lstm_cell_40/add:z:09lstm_40/while/lstm_cell_40/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������l
*lstm_40/while/lstm_cell_40/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
 lstm_40/while/lstm_cell_40/splitSplit3lstm_40/while/lstm_cell_40/split/split_dim:output:0+lstm_40/while/lstm_cell_40/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������d:���������d:���������d:���������d*
	num_split�
"lstm_40/while/lstm_cell_40/SigmoidSigmoid)lstm_40/while/lstm_cell_40/split:output:0*
T0*'
_output_shapes
:���������d�
$lstm_40/while/lstm_cell_40/Sigmoid_1Sigmoid)lstm_40/while/lstm_cell_40/split:output:1*
T0*'
_output_shapes
:���������d�
lstm_40/while/lstm_cell_40/mulMul(lstm_40/while/lstm_cell_40/Sigmoid_1:y:0lstm_40_while_placeholder_3*
T0*'
_output_shapes
:���������d�
lstm_40/while/lstm_cell_40/ReluRelu)lstm_40/while/lstm_cell_40/split:output:2*
T0*'
_output_shapes
:���������d�
 lstm_40/while/lstm_cell_40/mul_1Mul&lstm_40/while/lstm_cell_40/Sigmoid:y:0-lstm_40/while/lstm_cell_40/Relu:activations:0*
T0*'
_output_shapes
:���������d�
 lstm_40/while/lstm_cell_40/add_1AddV2"lstm_40/while/lstm_cell_40/mul:z:0$lstm_40/while/lstm_cell_40/mul_1:z:0*
T0*'
_output_shapes
:���������d�
$lstm_40/while/lstm_cell_40/Sigmoid_2Sigmoid)lstm_40/while/lstm_cell_40/split:output:3*
T0*'
_output_shapes
:���������d�
!lstm_40/while/lstm_cell_40/Relu_1Relu$lstm_40/while/lstm_cell_40/add_1:z:0*
T0*'
_output_shapes
:���������d�
 lstm_40/while/lstm_cell_40/mul_2Mul(lstm_40/while/lstm_cell_40/Sigmoid_2:y:0/lstm_40/while/lstm_cell_40/Relu_1:activations:0*
T0*'
_output_shapes
:���������dz
8lstm_40/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : �
2lstm_40/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_40_while_placeholder_1Alstm_40/while/TensorArrayV2Write/TensorListSetItem/index:output:0$lstm_40/while/lstm_cell_40/mul_2:z:0*
_output_shapes
: *
element_dtype0:���U
lstm_40/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :t
lstm_40/while/addAddV2lstm_40_while_placeholderlstm_40/while/add/y:output:0*
T0*
_output_shapes
: W
lstm_40/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_40/while/add_1AddV2(lstm_40_while_lstm_40_while_loop_counterlstm_40/while/add_1/y:output:0*
T0*
_output_shapes
: q
lstm_40/while/IdentityIdentitylstm_40/while/add_1:z:0^lstm_40/while/NoOp*
T0*
_output_shapes
: �
lstm_40/while/Identity_1Identity.lstm_40_while_lstm_40_while_maximum_iterations^lstm_40/while/NoOp*
T0*
_output_shapes
: q
lstm_40/while/Identity_2Identitylstm_40/while/add:z:0^lstm_40/while/NoOp*
T0*
_output_shapes
: �
lstm_40/while/Identity_3IdentityBlstm_40/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_40/while/NoOp*
T0*
_output_shapes
: �
lstm_40/while/Identity_4Identity$lstm_40/while/lstm_cell_40/mul_2:z:0^lstm_40/while/NoOp*
T0*'
_output_shapes
:���������d�
lstm_40/while/Identity_5Identity$lstm_40/while/lstm_cell_40/add_1:z:0^lstm_40/while/NoOp*
T0*'
_output_shapes
:���������d�
lstm_40/while/NoOpNoOp2^lstm_40/while/lstm_cell_40/BiasAdd/ReadVariableOp1^lstm_40/while/lstm_cell_40/MatMul/ReadVariableOp3^lstm_40/while/lstm_cell_40/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "9
lstm_40_while_identitylstm_40/while/Identity:output:0"=
lstm_40_while_identity_1!lstm_40/while/Identity_1:output:0"=
lstm_40_while_identity_2!lstm_40/while/Identity_2:output:0"=
lstm_40_while_identity_3!lstm_40/while/Identity_3:output:0"=
lstm_40_while_identity_4!lstm_40/while/Identity_4:output:0"=
lstm_40_while_identity_5!lstm_40/while/Identity_5:output:0"P
%lstm_40_while_lstm_40_strided_slice_1'lstm_40_while_lstm_40_strided_slice_1_0"z
:lstm_40_while_lstm_cell_40_biasadd_readvariableop_resource<lstm_40_while_lstm_cell_40_biasadd_readvariableop_resource_0"|
;lstm_40_while_lstm_cell_40_matmul_1_readvariableop_resource=lstm_40_while_lstm_cell_40_matmul_1_readvariableop_resource_0"x
9lstm_40_while_lstm_cell_40_matmul_readvariableop_resource;lstm_40_while_lstm_cell_40_matmul_readvariableop_resource_0"�
alstm_40_while_tensorarrayv2read_tensorlistgetitem_lstm_40_tensorarrayunstack_tensorlistfromtensorclstm_40_while_tensorarrayv2read_tensorlistgetitem_lstm_40_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������d:���������d: : : : : 2f
1lstm_40/while/lstm_cell_40/BiasAdd/ReadVariableOp1lstm_40/while/lstm_cell_40/BiasAdd/ReadVariableOp2d
0lstm_40/while/lstm_cell_40/MatMul/ReadVariableOp0lstm_40/while/lstm_cell_40/MatMul/ReadVariableOp2h
2lstm_40/while/lstm_cell_40/MatMul_1/ReadVariableOp2lstm_40/while/lstm_cell_40/MatMul_1/ReadVariableOp: 
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
�S
�
(sequential_40_lstm_40_while_body_1377220H
Dsequential_40_lstm_40_while_sequential_40_lstm_40_while_loop_counterN
Jsequential_40_lstm_40_while_sequential_40_lstm_40_while_maximum_iterations+
'sequential_40_lstm_40_while_placeholder-
)sequential_40_lstm_40_while_placeholder_1-
)sequential_40_lstm_40_while_placeholder_2-
)sequential_40_lstm_40_while_placeholder_3G
Csequential_40_lstm_40_while_sequential_40_lstm_40_strided_slice_1_0�
sequential_40_lstm_40_while_tensorarrayv2read_tensorlistgetitem_sequential_40_lstm_40_tensorarrayunstack_tensorlistfromtensor_0\
Isequential_40_lstm_40_while_lstm_cell_40_matmul_readvariableop_resource_0:	�^
Ksequential_40_lstm_40_while_lstm_cell_40_matmul_1_readvariableop_resource_0:	d�Y
Jsequential_40_lstm_40_while_lstm_cell_40_biasadd_readvariableop_resource_0:	�(
$sequential_40_lstm_40_while_identity*
&sequential_40_lstm_40_while_identity_1*
&sequential_40_lstm_40_while_identity_2*
&sequential_40_lstm_40_while_identity_3*
&sequential_40_lstm_40_while_identity_4*
&sequential_40_lstm_40_while_identity_5E
Asequential_40_lstm_40_while_sequential_40_lstm_40_strided_slice_1�
}sequential_40_lstm_40_while_tensorarrayv2read_tensorlistgetitem_sequential_40_lstm_40_tensorarrayunstack_tensorlistfromtensorZ
Gsequential_40_lstm_40_while_lstm_cell_40_matmul_readvariableop_resource:	�\
Isequential_40_lstm_40_while_lstm_cell_40_matmul_1_readvariableop_resource:	d�W
Hsequential_40_lstm_40_while_lstm_cell_40_biasadd_readvariableop_resource:	���?sequential_40/lstm_40/while/lstm_cell_40/BiasAdd/ReadVariableOp�>sequential_40/lstm_40/while/lstm_cell_40/MatMul/ReadVariableOp�@sequential_40/lstm_40/while/lstm_cell_40/MatMul_1/ReadVariableOp�
Msequential_40/lstm_40/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
?sequential_40/lstm_40/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsequential_40_lstm_40_while_tensorarrayv2read_tensorlistgetitem_sequential_40_lstm_40_tensorarrayunstack_tensorlistfromtensor_0'sequential_40_lstm_40_while_placeholderVsequential_40/lstm_40/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
>sequential_40/lstm_40/while/lstm_cell_40/MatMul/ReadVariableOpReadVariableOpIsequential_40_lstm_40_while_lstm_cell_40_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
/sequential_40/lstm_40/while/lstm_cell_40/MatMulMatMulFsequential_40/lstm_40/while/TensorArrayV2Read/TensorListGetItem:item:0Fsequential_40/lstm_40/while/lstm_cell_40/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
@sequential_40/lstm_40/while/lstm_cell_40/MatMul_1/ReadVariableOpReadVariableOpKsequential_40_lstm_40_while_lstm_cell_40_matmul_1_readvariableop_resource_0*
_output_shapes
:	d�*
dtype0�
1sequential_40/lstm_40/while/lstm_cell_40/MatMul_1MatMul)sequential_40_lstm_40_while_placeholder_2Hsequential_40/lstm_40/while/lstm_cell_40/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
,sequential_40/lstm_40/while/lstm_cell_40/addAddV29sequential_40/lstm_40/while/lstm_cell_40/MatMul:product:0;sequential_40/lstm_40/while/lstm_cell_40/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
?sequential_40/lstm_40/while/lstm_cell_40/BiasAdd/ReadVariableOpReadVariableOpJsequential_40_lstm_40_while_lstm_cell_40_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype0�
0sequential_40/lstm_40/while/lstm_cell_40/BiasAddBiasAdd0sequential_40/lstm_40/while/lstm_cell_40/add:z:0Gsequential_40/lstm_40/while/lstm_cell_40/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������z
8sequential_40/lstm_40/while/lstm_cell_40/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
.sequential_40/lstm_40/while/lstm_cell_40/splitSplitAsequential_40/lstm_40/while/lstm_cell_40/split/split_dim:output:09sequential_40/lstm_40/while/lstm_cell_40/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������d:���������d:���������d:���������d*
	num_split�
0sequential_40/lstm_40/while/lstm_cell_40/SigmoidSigmoid7sequential_40/lstm_40/while/lstm_cell_40/split:output:0*
T0*'
_output_shapes
:���������d�
2sequential_40/lstm_40/while/lstm_cell_40/Sigmoid_1Sigmoid7sequential_40/lstm_40/while/lstm_cell_40/split:output:1*
T0*'
_output_shapes
:���������d�
,sequential_40/lstm_40/while/lstm_cell_40/mulMul6sequential_40/lstm_40/while/lstm_cell_40/Sigmoid_1:y:0)sequential_40_lstm_40_while_placeholder_3*
T0*'
_output_shapes
:���������d�
-sequential_40/lstm_40/while/lstm_cell_40/ReluRelu7sequential_40/lstm_40/while/lstm_cell_40/split:output:2*
T0*'
_output_shapes
:���������d�
.sequential_40/lstm_40/while/lstm_cell_40/mul_1Mul4sequential_40/lstm_40/while/lstm_cell_40/Sigmoid:y:0;sequential_40/lstm_40/while/lstm_cell_40/Relu:activations:0*
T0*'
_output_shapes
:���������d�
.sequential_40/lstm_40/while/lstm_cell_40/add_1AddV20sequential_40/lstm_40/while/lstm_cell_40/mul:z:02sequential_40/lstm_40/while/lstm_cell_40/mul_1:z:0*
T0*'
_output_shapes
:���������d�
2sequential_40/lstm_40/while/lstm_cell_40/Sigmoid_2Sigmoid7sequential_40/lstm_40/while/lstm_cell_40/split:output:3*
T0*'
_output_shapes
:���������d�
/sequential_40/lstm_40/while/lstm_cell_40/Relu_1Relu2sequential_40/lstm_40/while/lstm_cell_40/add_1:z:0*
T0*'
_output_shapes
:���������d�
.sequential_40/lstm_40/while/lstm_cell_40/mul_2Mul6sequential_40/lstm_40/while/lstm_cell_40/Sigmoid_2:y:0=sequential_40/lstm_40/while/lstm_cell_40/Relu_1:activations:0*
T0*'
_output_shapes
:���������d�
Fsequential_40/lstm_40/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : �
@sequential_40/lstm_40/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem)sequential_40_lstm_40_while_placeholder_1Osequential_40/lstm_40/while/TensorArrayV2Write/TensorListSetItem/index:output:02sequential_40/lstm_40/while/lstm_cell_40/mul_2:z:0*
_output_shapes
: *
element_dtype0:���c
!sequential_40/lstm_40/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :�
sequential_40/lstm_40/while/addAddV2'sequential_40_lstm_40_while_placeholder*sequential_40/lstm_40/while/add/y:output:0*
T0*
_output_shapes
: e
#sequential_40/lstm_40/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
!sequential_40/lstm_40/while/add_1AddV2Dsequential_40_lstm_40_while_sequential_40_lstm_40_while_loop_counter,sequential_40/lstm_40/while/add_1/y:output:0*
T0*
_output_shapes
: �
$sequential_40/lstm_40/while/IdentityIdentity%sequential_40/lstm_40/while/add_1:z:0!^sequential_40/lstm_40/while/NoOp*
T0*
_output_shapes
: �
&sequential_40/lstm_40/while/Identity_1IdentityJsequential_40_lstm_40_while_sequential_40_lstm_40_while_maximum_iterations!^sequential_40/lstm_40/while/NoOp*
T0*
_output_shapes
: �
&sequential_40/lstm_40/while/Identity_2Identity#sequential_40/lstm_40/while/add:z:0!^sequential_40/lstm_40/while/NoOp*
T0*
_output_shapes
: �
&sequential_40/lstm_40/while/Identity_3IdentityPsequential_40/lstm_40/while/TensorArrayV2Write/TensorListSetItem:output_handle:0!^sequential_40/lstm_40/while/NoOp*
T0*
_output_shapes
: �
&sequential_40/lstm_40/while/Identity_4Identity2sequential_40/lstm_40/while/lstm_cell_40/mul_2:z:0!^sequential_40/lstm_40/while/NoOp*
T0*'
_output_shapes
:���������d�
&sequential_40/lstm_40/while/Identity_5Identity2sequential_40/lstm_40/while/lstm_cell_40/add_1:z:0!^sequential_40/lstm_40/while/NoOp*
T0*'
_output_shapes
:���������d�
 sequential_40/lstm_40/while/NoOpNoOp@^sequential_40/lstm_40/while/lstm_cell_40/BiasAdd/ReadVariableOp?^sequential_40/lstm_40/while/lstm_cell_40/MatMul/ReadVariableOpA^sequential_40/lstm_40/while/lstm_cell_40/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "U
$sequential_40_lstm_40_while_identity-sequential_40/lstm_40/while/Identity:output:0"Y
&sequential_40_lstm_40_while_identity_1/sequential_40/lstm_40/while/Identity_1:output:0"Y
&sequential_40_lstm_40_while_identity_2/sequential_40/lstm_40/while/Identity_2:output:0"Y
&sequential_40_lstm_40_while_identity_3/sequential_40/lstm_40/while/Identity_3:output:0"Y
&sequential_40_lstm_40_while_identity_4/sequential_40/lstm_40/while/Identity_4:output:0"Y
&sequential_40_lstm_40_while_identity_5/sequential_40/lstm_40/while/Identity_5:output:0"�
Hsequential_40_lstm_40_while_lstm_cell_40_biasadd_readvariableop_resourceJsequential_40_lstm_40_while_lstm_cell_40_biasadd_readvariableop_resource_0"�
Isequential_40_lstm_40_while_lstm_cell_40_matmul_1_readvariableop_resourceKsequential_40_lstm_40_while_lstm_cell_40_matmul_1_readvariableop_resource_0"�
Gsequential_40_lstm_40_while_lstm_cell_40_matmul_readvariableop_resourceIsequential_40_lstm_40_while_lstm_cell_40_matmul_readvariableop_resource_0"�
Asequential_40_lstm_40_while_sequential_40_lstm_40_strided_slice_1Csequential_40_lstm_40_while_sequential_40_lstm_40_strided_slice_1_0"�
}sequential_40_lstm_40_while_tensorarrayv2read_tensorlistgetitem_sequential_40_lstm_40_tensorarrayunstack_tensorlistfromtensorsequential_40_lstm_40_while_tensorarrayv2read_tensorlistgetitem_sequential_40_lstm_40_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������d:���������d: : : : : 2�
?sequential_40/lstm_40/while/lstm_cell_40/BiasAdd/ReadVariableOp?sequential_40/lstm_40/while/lstm_cell_40/BiasAdd/ReadVariableOp2�
>sequential_40/lstm_40/while/lstm_cell_40/MatMul/ReadVariableOp>sequential_40/lstm_40/while/lstm_cell_40/MatMul/ReadVariableOp2�
@sequential_40/lstm_40/while/lstm_cell_40/MatMul_1/ReadVariableOp@sequential_40/lstm_40/while/lstm_cell_40/MatMul_1/ReadVariableOp: 
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
�
�
J__inference_sequential_40_layer_call_and_return_conditional_losses_1378228
lstm_40_input"
lstm_40_1378205:	�"
lstm_40_1378207:	d�
lstm_40_1378209:	�#
dense_120_1378212:d2
dense_120_1378214:2#
dense_121_1378217:22
dense_121_1378219:2#
dense_122_1378222:2
dense_122_1378224:
identity��!dense_120/StatefulPartitionedCall�!dense_121/StatefulPartitionedCall�!dense_122/StatefulPartitionedCall�lstm_40/StatefulPartitionedCall�
lstm_40/StatefulPartitionedCallStatefulPartitionedCalllstm_40_inputlstm_40_1378205lstm_40_1378207lstm_40_1378209*
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
GPU 2J 8� *M
fHRF
D__inference_lstm_40_layer_call_and_return_conditional_losses_1377830�
!dense_120/StatefulPartitionedCallStatefulPartitionedCall(lstm_40/StatefulPartitionedCall:output:0dense_120_1378212dense_120_1378214*
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
GPU 2J 8� *O
fJRH
F__inference_dense_120_layer_call_and_return_conditional_losses_1377849�
!dense_121/StatefulPartitionedCallStatefulPartitionedCall*dense_120/StatefulPartitionedCall:output:0dense_121_1378217dense_121_1378219*
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
GPU 2J 8� *O
fJRH
F__inference_dense_121_layer_call_and_return_conditional_losses_1377866�
!dense_122/StatefulPartitionedCallStatefulPartitionedCall*dense_121/StatefulPartitionedCall:output:0dense_122_1378222dense_122_1378224*
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
GPU 2J 8� *O
fJRH
F__inference_dense_122_layer_call_and_return_conditional_losses_1377882y
IdentityIdentity*dense_122/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp"^dense_120/StatefulPartitionedCall"^dense_121/StatefulPartitionedCall"^dense_122/StatefulPartitionedCall ^lstm_40/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):���������: : : : : : : : : 2F
!dense_120/StatefulPartitionedCall!dense_120/StatefulPartitionedCall2F
!dense_121/StatefulPartitionedCall!dense_121/StatefulPartitionedCall2F
!dense_122/StatefulPartitionedCall!dense_122/StatefulPartitionedCall2B
lstm_40/StatefulPartitionedCalllstm_40/StatefulPartitionedCall:Z V
+
_output_shapes
:���������
'
_user_specified_namelstm_40_input
�
�
while_cond_1377744
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1377744___redundant_placeholder05
1while_while_cond_1377744___redundant_placeholder15
1while_while_cond_1377744___redundant_placeholder25
1while_while_cond_1377744___redundant_placeholder3
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

�
lstm_40_while_cond_1378390,
(lstm_40_while_lstm_40_while_loop_counter2
.lstm_40_while_lstm_40_while_maximum_iterations
lstm_40_while_placeholder
lstm_40_while_placeholder_1
lstm_40_while_placeholder_2
lstm_40_while_placeholder_3.
*lstm_40_while_less_lstm_40_strided_slice_1E
Alstm_40_while_lstm_40_while_cond_1378390___redundant_placeholder0E
Alstm_40_while_lstm_40_while_cond_1378390___redundant_placeholder1E
Alstm_40_while_lstm_40_while_cond_1378390___redundant_placeholder2E
Alstm_40_while_lstm_40_while_cond_1378390___redundant_placeholder3
lstm_40_while_identity
�
lstm_40/while/LessLesslstm_40_while_placeholder*lstm_40_while_less_lstm_40_strided_slice_1*
T0*
_output_shapes
: [
lstm_40/while/IdentityIdentitylstm_40/while/Less:z:0*
T0
*
_output_shapes
: "9
lstm_40_while_identitylstm_40/while/Identity:output:0*(
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
/__inference_sequential_40_layer_call_fn_1378308

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
GPU 2J 8� *S
fNRL
J__inference_sequential_40_layer_call_and_return_conditional_losses_1377889o
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

�
/__inference_sequential_40_layer_call_fn_1378331

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
GPU 2J 8� *S
fNRL
J__inference_sequential_40_layer_call_and_return_conditional_losses_1378158o
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
�
�
J__inference_sequential_40_layer_call_and_return_conditional_losses_1378254
lstm_40_input"
lstm_40_1378231:	�"
lstm_40_1378233:	d�
lstm_40_1378235:	�#
dense_120_1378238:d2
dense_120_1378240:2#
dense_121_1378243:22
dense_121_1378245:2#
dense_122_1378248:2
dense_122_1378250:
identity��!dense_120/StatefulPartitionedCall�!dense_121/StatefulPartitionedCall�!dense_122/StatefulPartitionedCall�lstm_40/StatefulPartitionedCall�
lstm_40/StatefulPartitionedCallStatefulPartitionedCalllstm_40_inputlstm_40_1378231lstm_40_1378233lstm_40_1378235*
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
GPU 2J 8� *M
fHRF
D__inference_lstm_40_layer_call_and_return_conditional_losses_1378098�
!dense_120/StatefulPartitionedCallStatefulPartitionedCall(lstm_40/StatefulPartitionedCall:output:0dense_120_1378238dense_120_1378240*
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
GPU 2J 8� *O
fJRH
F__inference_dense_120_layer_call_and_return_conditional_losses_1377849�
!dense_121/StatefulPartitionedCallStatefulPartitionedCall*dense_120/StatefulPartitionedCall:output:0dense_121_1378243dense_121_1378245*
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
GPU 2J 8� *O
fJRH
F__inference_dense_121_layer_call_and_return_conditional_losses_1377866�
!dense_122/StatefulPartitionedCallStatefulPartitionedCall*dense_121/StatefulPartitionedCall:output:0dense_122_1378248dense_122_1378250*
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
GPU 2J 8� *O
fJRH
F__inference_dense_122_layer_call_and_return_conditional_losses_1377882y
IdentityIdentity*dense_122/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp"^dense_120/StatefulPartitionedCall"^dense_121/StatefulPartitionedCall"^dense_122/StatefulPartitionedCall ^lstm_40/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):���������: : : : : : : : : 2F
!dense_120/StatefulPartitionedCall!dense_120/StatefulPartitionedCall2F
!dense_121/StatefulPartitionedCall!dense_121/StatefulPartitionedCall2F
!dense_122/StatefulPartitionedCall!dense_122/StatefulPartitionedCall2B
lstm_40/StatefulPartitionedCalllstm_40/StatefulPartitionedCall:Z V
+
_output_shapes
:���������
'
_user_specified_namelstm_40_input
�
�
)__inference_lstm_40_layer_call_fn_1378705

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
GPU 2J 8� *M
fHRF
D__inference_lstm_40_layer_call_and_return_conditional_losses_1378098o
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
�B
�

lstm_40_while_body_1378391,
(lstm_40_while_lstm_40_while_loop_counter2
.lstm_40_while_lstm_40_while_maximum_iterations
lstm_40_while_placeholder
lstm_40_while_placeholder_1
lstm_40_while_placeholder_2
lstm_40_while_placeholder_3+
'lstm_40_while_lstm_40_strided_slice_1_0g
clstm_40_while_tensorarrayv2read_tensorlistgetitem_lstm_40_tensorarrayunstack_tensorlistfromtensor_0N
;lstm_40_while_lstm_cell_40_matmul_readvariableop_resource_0:	�P
=lstm_40_while_lstm_cell_40_matmul_1_readvariableop_resource_0:	d�K
<lstm_40_while_lstm_cell_40_biasadd_readvariableop_resource_0:	�
lstm_40_while_identity
lstm_40_while_identity_1
lstm_40_while_identity_2
lstm_40_while_identity_3
lstm_40_while_identity_4
lstm_40_while_identity_5)
%lstm_40_while_lstm_40_strided_slice_1e
alstm_40_while_tensorarrayv2read_tensorlistgetitem_lstm_40_tensorarrayunstack_tensorlistfromtensorL
9lstm_40_while_lstm_cell_40_matmul_readvariableop_resource:	�N
;lstm_40_while_lstm_cell_40_matmul_1_readvariableop_resource:	d�I
:lstm_40_while_lstm_cell_40_biasadd_readvariableop_resource:	���1lstm_40/while/lstm_cell_40/BiasAdd/ReadVariableOp�0lstm_40/while/lstm_cell_40/MatMul/ReadVariableOp�2lstm_40/while/lstm_cell_40/MatMul_1/ReadVariableOp�
?lstm_40/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
1lstm_40/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_40_while_tensorarrayv2read_tensorlistgetitem_lstm_40_tensorarrayunstack_tensorlistfromtensor_0lstm_40_while_placeholderHlstm_40/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
0lstm_40/while/lstm_cell_40/MatMul/ReadVariableOpReadVariableOp;lstm_40_while_lstm_cell_40_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
!lstm_40/while/lstm_cell_40/MatMulMatMul8lstm_40/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_40/while/lstm_cell_40/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
2lstm_40/while/lstm_cell_40/MatMul_1/ReadVariableOpReadVariableOp=lstm_40_while_lstm_cell_40_matmul_1_readvariableop_resource_0*
_output_shapes
:	d�*
dtype0�
#lstm_40/while/lstm_cell_40/MatMul_1MatMullstm_40_while_placeholder_2:lstm_40/while/lstm_cell_40/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
lstm_40/while/lstm_cell_40/addAddV2+lstm_40/while/lstm_cell_40/MatMul:product:0-lstm_40/while/lstm_cell_40/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
1lstm_40/while/lstm_cell_40/BiasAdd/ReadVariableOpReadVariableOp<lstm_40_while_lstm_cell_40_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype0�
"lstm_40/while/lstm_cell_40/BiasAddBiasAdd"lstm_40/while/lstm_cell_40/add:z:09lstm_40/while/lstm_cell_40/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������l
*lstm_40/while/lstm_cell_40/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
 lstm_40/while/lstm_cell_40/splitSplit3lstm_40/while/lstm_cell_40/split/split_dim:output:0+lstm_40/while/lstm_cell_40/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������d:���������d:���������d:���������d*
	num_split�
"lstm_40/while/lstm_cell_40/SigmoidSigmoid)lstm_40/while/lstm_cell_40/split:output:0*
T0*'
_output_shapes
:���������d�
$lstm_40/while/lstm_cell_40/Sigmoid_1Sigmoid)lstm_40/while/lstm_cell_40/split:output:1*
T0*'
_output_shapes
:���������d�
lstm_40/while/lstm_cell_40/mulMul(lstm_40/while/lstm_cell_40/Sigmoid_1:y:0lstm_40_while_placeholder_3*
T0*'
_output_shapes
:���������d�
lstm_40/while/lstm_cell_40/ReluRelu)lstm_40/while/lstm_cell_40/split:output:2*
T0*'
_output_shapes
:���������d�
 lstm_40/while/lstm_cell_40/mul_1Mul&lstm_40/while/lstm_cell_40/Sigmoid:y:0-lstm_40/while/lstm_cell_40/Relu:activations:0*
T0*'
_output_shapes
:���������d�
 lstm_40/while/lstm_cell_40/add_1AddV2"lstm_40/while/lstm_cell_40/mul:z:0$lstm_40/while/lstm_cell_40/mul_1:z:0*
T0*'
_output_shapes
:���������d�
$lstm_40/while/lstm_cell_40/Sigmoid_2Sigmoid)lstm_40/while/lstm_cell_40/split:output:3*
T0*'
_output_shapes
:���������d�
!lstm_40/while/lstm_cell_40/Relu_1Relu$lstm_40/while/lstm_cell_40/add_1:z:0*
T0*'
_output_shapes
:���������d�
 lstm_40/while/lstm_cell_40/mul_2Mul(lstm_40/while/lstm_cell_40/Sigmoid_2:y:0/lstm_40/while/lstm_cell_40/Relu_1:activations:0*
T0*'
_output_shapes
:���������dz
8lstm_40/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : �
2lstm_40/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_40_while_placeholder_1Alstm_40/while/TensorArrayV2Write/TensorListSetItem/index:output:0$lstm_40/while/lstm_cell_40/mul_2:z:0*
_output_shapes
: *
element_dtype0:���U
lstm_40/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :t
lstm_40/while/addAddV2lstm_40_while_placeholderlstm_40/while/add/y:output:0*
T0*
_output_shapes
: W
lstm_40/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_40/while/add_1AddV2(lstm_40_while_lstm_40_while_loop_counterlstm_40/while/add_1/y:output:0*
T0*
_output_shapes
: q
lstm_40/while/IdentityIdentitylstm_40/while/add_1:z:0^lstm_40/while/NoOp*
T0*
_output_shapes
: �
lstm_40/while/Identity_1Identity.lstm_40_while_lstm_40_while_maximum_iterations^lstm_40/while/NoOp*
T0*
_output_shapes
: q
lstm_40/while/Identity_2Identitylstm_40/while/add:z:0^lstm_40/while/NoOp*
T0*
_output_shapes
: �
lstm_40/while/Identity_3IdentityBlstm_40/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_40/while/NoOp*
T0*
_output_shapes
: �
lstm_40/while/Identity_4Identity$lstm_40/while/lstm_cell_40/mul_2:z:0^lstm_40/while/NoOp*
T0*'
_output_shapes
:���������d�
lstm_40/while/Identity_5Identity$lstm_40/while/lstm_cell_40/add_1:z:0^lstm_40/while/NoOp*
T0*'
_output_shapes
:���������d�
lstm_40/while/NoOpNoOp2^lstm_40/while/lstm_cell_40/BiasAdd/ReadVariableOp1^lstm_40/while/lstm_cell_40/MatMul/ReadVariableOp3^lstm_40/while/lstm_cell_40/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "9
lstm_40_while_identitylstm_40/while/Identity:output:0"=
lstm_40_while_identity_1!lstm_40/while/Identity_1:output:0"=
lstm_40_while_identity_2!lstm_40/while/Identity_2:output:0"=
lstm_40_while_identity_3!lstm_40/while/Identity_3:output:0"=
lstm_40_while_identity_4!lstm_40/while/Identity_4:output:0"=
lstm_40_while_identity_5!lstm_40/while/Identity_5:output:0"P
%lstm_40_while_lstm_40_strided_slice_1'lstm_40_while_lstm_40_strided_slice_1_0"z
:lstm_40_while_lstm_cell_40_biasadd_readvariableop_resource<lstm_40_while_lstm_cell_40_biasadd_readvariableop_resource_0"|
;lstm_40_while_lstm_cell_40_matmul_1_readvariableop_resource=lstm_40_while_lstm_cell_40_matmul_1_readvariableop_resource_0"x
9lstm_40_while_lstm_cell_40_matmul_readvariableop_resource;lstm_40_while_lstm_cell_40_matmul_readvariableop_resource_0"�
alstm_40_while_tensorarrayv2read_tensorlistgetitem_lstm_40_tensorarrayunstack_tensorlistfromtensorclstm_40_while_tensorarrayv2read_tensorlistgetitem_lstm_40_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������d:���������d: : : : : 2f
1lstm_40/while/lstm_cell_40/BiasAdd/ReadVariableOp1lstm_40/while/lstm_cell_40/BiasAdd/ReadVariableOp2d
0lstm_40/while/lstm_cell_40/MatMul/ReadVariableOp0lstm_40/while/lstm_cell_40/MatMul/ReadVariableOp2h
2lstm_40/while/lstm_cell_40/MatMul_1/ReadVariableOp2lstm_40/while/lstm_cell_40/MatMul_1/ReadVariableOp: 
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
I__inference_lstm_cell_40_layer_call_and_return_conditional_losses_1379442

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
�K
�
D__inference_lstm_40_layer_call_and_return_conditional_losses_1377830

inputs>
+lstm_cell_40_matmul_readvariableop_resource:	�@
-lstm_cell_40_matmul_1_readvariableop_resource:	d�;
,lstm_cell_40_biasadd_readvariableop_resource:	�
identity��#lstm_cell_40/BiasAdd/ReadVariableOp�"lstm_cell_40/MatMul/ReadVariableOp�$lstm_cell_40/MatMul_1/ReadVariableOp�while;
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
"lstm_cell_40/MatMul/ReadVariableOpReadVariableOp+lstm_cell_40_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
lstm_cell_40/MatMulMatMulstrided_slice_2:output:0*lstm_cell_40/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
$lstm_cell_40/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_40_matmul_1_readvariableop_resource*
_output_shapes
:	d�*
dtype0�
lstm_cell_40/MatMul_1MatMulzeros:output:0,lstm_cell_40/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
lstm_cell_40/addAddV2lstm_cell_40/MatMul:product:0lstm_cell_40/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
#lstm_cell_40/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_40_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
lstm_cell_40/BiasAddBiasAddlstm_cell_40/add:z:0+lstm_cell_40/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������^
lstm_cell_40/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_cell_40/splitSplit%lstm_cell_40/split/split_dim:output:0lstm_cell_40/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������d:���������d:���������d:���������d*
	num_splitn
lstm_cell_40/SigmoidSigmoidlstm_cell_40/split:output:0*
T0*'
_output_shapes
:���������dp
lstm_cell_40/Sigmoid_1Sigmoidlstm_cell_40/split:output:1*
T0*'
_output_shapes
:���������dw
lstm_cell_40/mulMullstm_cell_40/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������dh
lstm_cell_40/ReluRelulstm_cell_40/split:output:2*
T0*'
_output_shapes
:���������d�
lstm_cell_40/mul_1Mullstm_cell_40/Sigmoid:y:0lstm_cell_40/Relu:activations:0*
T0*'
_output_shapes
:���������d{
lstm_cell_40/add_1AddV2lstm_cell_40/mul:z:0lstm_cell_40/mul_1:z:0*
T0*'
_output_shapes
:���������dp
lstm_cell_40/Sigmoid_2Sigmoidlstm_cell_40/split:output:3*
T0*'
_output_shapes
:���������de
lstm_cell_40/Relu_1Relulstm_cell_40/add_1:z:0*
T0*'
_output_shapes
:���������d�
lstm_cell_40/mul_2Mullstm_cell_40/Sigmoid_2:y:0!lstm_cell_40/Relu_1:activations:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_40_matmul_readvariableop_resource-lstm_cell_40_matmul_1_readvariableop_resource,lstm_cell_40_biasadd_readvariableop_resource*
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
bodyR
while_body_1377745*
condR
while_cond_1377744*K
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
NoOpNoOp$^lstm_cell_40/BiasAdd/ReadVariableOp#^lstm_cell_40/MatMul/ReadVariableOp%^lstm_cell_40/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������: : : 2J
#lstm_cell_40/BiasAdd/ReadVariableOp#lstm_cell_40/BiasAdd/ReadVariableOp2H
"lstm_cell_40/MatMul/ReadVariableOp"lstm_cell_40/MatMul/ReadVariableOp2L
$lstm_cell_40/MatMul_1/ReadVariableOp$lstm_cell_40/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:���������
 
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
lstm_40_input:
serving_default_lstm_40_input:0���������=
	dense_1220
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
/__inference_sequential_40_layer_call_fn_1377910
/__inference_sequential_40_layer_call_fn_1378308
/__inference_sequential_40_layer_call_fn_1378331
/__inference_sequential_40_layer_call_fn_1378202�
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
J__inference_sequential_40_layer_call_and_return_conditional_losses_1378496
J__inference_sequential_40_layer_call_and_return_conditional_losses_1378661
J__inference_sequential_40_layer_call_and_return_conditional_losses_1378228
J__inference_sequential_40_layer_call_and_return_conditional_losses_1378254�
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
"__inference__wrapped_model_1377325lstm_40_input"�
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
)__inference_lstm_40_layer_call_fn_1378672
)__inference_lstm_40_layer_call_fn_1378683
)__inference_lstm_40_layer_call_fn_1378694
)__inference_lstm_40_layer_call_fn_1378705�
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
D__inference_lstm_40_layer_call_and_return_conditional_losses_1378850
D__inference_lstm_40_layer_call_and_return_conditional_losses_1378995
D__inference_lstm_40_layer_call_and_return_conditional_losses_1379140
D__inference_lstm_40_layer_call_and_return_conditional_losses_1379285�
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
+__inference_dense_120_layer_call_fn_1379294�
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
F__inference_dense_120_layer_call_and_return_conditional_losses_1379305�
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
": d22dense_120/kernel
:22dense_120/bias
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
+__inference_dense_121_layer_call_fn_1379314�
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
F__inference_dense_121_layer_call_and_return_conditional_losses_1379325�
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
": 222dense_121/kernel
:22dense_121/bias
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
+__inference_dense_122_layer_call_fn_1379334�
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
F__inference_dense_122_layer_call_and_return_conditional_losses_1379344�
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
": 22dense_122/kernel
:2dense_122/bias
.:,	�2lstm_40/lstm_cell_40/kernel
8:6	d�2%lstm_40/lstm_cell_40/recurrent_kernel
(:&�2lstm_40/lstm_cell_40/bias
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
/__inference_sequential_40_layer_call_fn_1377910lstm_40_input"�
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
/__inference_sequential_40_layer_call_fn_1378308inputs"�
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
/__inference_sequential_40_layer_call_fn_1378331inputs"�
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
/__inference_sequential_40_layer_call_fn_1378202lstm_40_input"�
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
J__inference_sequential_40_layer_call_and_return_conditional_losses_1378496inputs"�
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
J__inference_sequential_40_layer_call_and_return_conditional_losses_1378661inputs"�
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
J__inference_sequential_40_layer_call_and_return_conditional_losses_1378228lstm_40_input"�
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
J__inference_sequential_40_layer_call_and_return_conditional_losses_1378254lstm_40_input"�
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
%__inference_signature_wrapper_1378285lstm_40_input"�
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
)__inference_lstm_40_layer_call_fn_1378672inputs/0"�
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
)__inference_lstm_40_layer_call_fn_1378683inputs/0"�
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
)__inference_lstm_40_layer_call_fn_1378694inputs"�
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
)__inference_lstm_40_layer_call_fn_1378705inputs"�
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
D__inference_lstm_40_layer_call_and_return_conditional_losses_1378850inputs/0"�
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
D__inference_lstm_40_layer_call_and_return_conditional_losses_1378995inputs/0"�
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
D__inference_lstm_40_layer_call_and_return_conditional_losses_1379140inputs"�
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
D__inference_lstm_40_layer_call_and_return_conditional_losses_1379285inputs"�
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
.__inference_lstm_cell_40_layer_call_fn_1379361
.__inference_lstm_cell_40_layer_call_fn_1379378�
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
I__inference_lstm_cell_40_layer_call_and_return_conditional_losses_1379410
I__inference_lstm_cell_40_layer_call_and_return_conditional_losses_1379442�
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
+__inference_dense_120_layer_call_fn_1379294inputs"�
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
F__inference_dense_120_layer_call_and_return_conditional_losses_1379305inputs"�
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
+__inference_dense_121_layer_call_fn_1379314inputs"�
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
F__inference_dense_121_layer_call_and_return_conditional_losses_1379325inputs"�
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
+__inference_dense_122_layer_call_fn_1379334inputs"�
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
F__inference_dense_122_layer_call_and_return_conditional_losses_1379344inputs"�
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
.__inference_lstm_cell_40_layer_call_fn_1379361inputsstates/0states/1"�
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
.__inference_lstm_cell_40_layer_call_fn_1379378inputsstates/0states/1"�
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
I__inference_lstm_cell_40_layer_call_and_return_conditional_losses_1379410inputsstates/0states/1"�
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
I__inference_lstm_cell_40_layer_call_and_return_conditional_losses_1379442inputsstates/0states/1"�
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
':%d22Adam/dense_120/kernel/m
!:22Adam/dense_120/bias/m
':%222Adam/dense_121/kernel/m
!:22Adam/dense_121/bias/m
':%22Adam/dense_122/kernel/m
!:2Adam/dense_122/bias/m
3:1	�2"Adam/lstm_40/lstm_cell_40/kernel/m
=:;	d�2,Adam/lstm_40/lstm_cell_40/recurrent_kernel/m
-:+�2 Adam/lstm_40/lstm_cell_40/bias/m
':%d22Adam/dense_120/kernel/v
!:22Adam/dense_120/bias/v
':%222Adam/dense_121/kernel/v
!:22Adam/dense_121/bias/v
':%22Adam/dense_122/kernel/v
!:2Adam/dense_122/bias/v
3:1	�2"Adam/lstm_40/lstm_cell_40/kernel/v
=:;	d�2,Adam/lstm_40/lstm_cell_40/recurrent_kernel/v
-:+�2 Adam/lstm_40/lstm_cell_40/bias/v�
"__inference__wrapped_model_1377325~	/01%&-.:�7
0�-
+�(
lstm_40_input���������
� "5�2
0
	dense_122#� 
	dense_122����������
F__inference_dense_120_layer_call_and_return_conditional_losses_1379305\/�,
%�"
 �
inputs���������d
� "%�"
�
0���������2
� ~
+__inference_dense_120_layer_call_fn_1379294O/�,
%�"
 �
inputs���������d
� "����������2�
F__inference_dense_121_layer_call_and_return_conditional_losses_1379325\%&/�,
%�"
 �
inputs���������2
� "%�"
�
0���������2
� ~
+__inference_dense_121_layer_call_fn_1379314O%&/�,
%�"
 �
inputs���������2
� "����������2�
F__inference_dense_122_layer_call_and_return_conditional_losses_1379344\-./�,
%�"
 �
inputs���������2
� "%�"
�
0���������
� ~
+__inference_dense_122_layer_call_fn_1379334O-./�,
%�"
 �
inputs���������2
� "�����������
D__inference_lstm_40_layer_call_and_return_conditional_losses_1378850}/01O�L
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
D__inference_lstm_40_layer_call_and_return_conditional_losses_1378995}/01O�L
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
D__inference_lstm_40_layer_call_and_return_conditional_losses_1379140m/01?�<
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
D__inference_lstm_40_layer_call_and_return_conditional_losses_1379285m/01?�<
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
)__inference_lstm_40_layer_call_fn_1378672p/01O�L
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
)__inference_lstm_40_layer_call_fn_1378683p/01O�L
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
)__inference_lstm_40_layer_call_fn_1378694`/01?�<
5�2
$�!
inputs���������

 
p 

 
� "����������d�
)__inference_lstm_40_layer_call_fn_1378705`/01?�<
5�2
$�!
inputs���������

 
p

 
� "����������d�
I__inference_lstm_cell_40_layer_call_and_return_conditional_losses_1379410�/01��}
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
I__inference_lstm_cell_40_layer_call_and_return_conditional_losses_1379442�/01��}
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
.__inference_lstm_cell_40_layer_call_fn_1379361�/01��}
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
.__inference_lstm_cell_40_layer_call_fn_1379378�/01��}
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
J__inference_sequential_40_layer_call_and_return_conditional_losses_1378228v	/01%&-.B�?
8�5
+�(
lstm_40_input���������
p 

 
� "%�"
�
0���������
� �
J__inference_sequential_40_layer_call_and_return_conditional_losses_1378254v	/01%&-.B�?
8�5
+�(
lstm_40_input���������
p

 
� "%�"
�
0���������
� �
J__inference_sequential_40_layer_call_and_return_conditional_losses_1378496o	/01%&-.;�8
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
J__inference_sequential_40_layer_call_and_return_conditional_losses_1378661o	/01%&-.;�8
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
/__inference_sequential_40_layer_call_fn_1377910i	/01%&-.B�?
8�5
+�(
lstm_40_input���������
p 

 
� "�����������
/__inference_sequential_40_layer_call_fn_1378202i	/01%&-.B�?
8�5
+�(
lstm_40_input���������
p

 
� "�����������
/__inference_sequential_40_layer_call_fn_1378308b	/01%&-.;�8
1�.
$�!
inputs���������
p 

 
� "�����������
/__inference_sequential_40_layer_call_fn_1378331b	/01%&-.;�8
1�.
$�!
inputs���������
p

 
� "�����������
%__inference_signature_wrapper_1378285�	/01%&-.K�H
� 
A�>
<
lstm_40_input+�(
lstm_40_input���������"5�2
0
	dense_122#� 
	dense_122���������