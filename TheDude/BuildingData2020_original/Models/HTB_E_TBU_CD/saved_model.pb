л
џЯ
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
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

MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( 
?
Mul
x"T
y"T
z"T"
Ttype:
2	
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
dtypetype
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
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
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
С
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
executor_typestring Ј
@
StaticRegexFullMatch	
input

output
"
patternstring
ї
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
А
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type/
output_handleщшelement_dtype"
element_dtypetype"

shape_typetype:
2	

TensorListReserve
element_shape"
shape_type
num_elements(
handleщшelement_dtype"
element_dtypetype"

shape_typetype:
2	

TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsintџџџџџџџџџ
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 

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
"serve*2.10.02v2.10.0-rc3-6-g359c3cdfc5f8Йў

 Adam/lstm_27/lstm_cell_27/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/lstm_27/lstm_cell_27/bias/v

4Adam/lstm_27/lstm_cell_27/bias/v/Read/ReadVariableOpReadVariableOp Adam/lstm_27/lstm_cell_27/bias/v*
_output_shapes	
:*
dtype0
Е
,Adam/lstm_27/lstm_cell_27/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	d*=
shared_name.,Adam/lstm_27/lstm_cell_27/recurrent_kernel/v
Ў
@Adam/lstm_27/lstm_cell_27/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp,Adam/lstm_27/lstm_cell_27/recurrent_kernel/v*
_output_shapes
:	d*
dtype0
Ё
"Adam/lstm_27/lstm_cell_27/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*3
shared_name$"Adam/lstm_27/lstm_cell_27/kernel/v

6Adam/lstm_27/lstm_cell_27/kernel/v/Read/ReadVariableOpReadVariableOp"Adam/lstm_27/lstm_cell_27/kernel/v*
_output_shapes
:	*
dtype0

Adam/dense_83/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_83/bias/v
y
(Adam/dense_83/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_83/bias/v*
_output_shapes
:*
dtype0

Adam/dense_83/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*'
shared_nameAdam/dense_83/kernel/v

*Adam/dense_83/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_83/kernel/v*
_output_shapes

:2*
dtype0

Adam/dense_82/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*%
shared_nameAdam/dense_82/bias/v
y
(Adam/dense_82/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_82/bias/v*
_output_shapes
:2*
dtype0

Adam/dense_82/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:22*'
shared_nameAdam/dense_82/kernel/v

*Adam/dense_82/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_82/kernel/v*
_output_shapes

:22*
dtype0

Adam/dense_81/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*%
shared_nameAdam/dense_81/bias/v
y
(Adam/dense_81/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_81/bias/v*
_output_shapes
:2*
dtype0

Adam/dense_81/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d2*'
shared_nameAdam/dense_81/kernel/v

*Adam/dense_81/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_81/kernel/v*
_output_shapes

:d2*
dtype0

 Adam/lstm_27/lstm_cell_27/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/lstm_27/lstm_cell_27/bias/m

4Adam/lstm_27/lstm_cell_27/bias/m/Read/ReadVariableOpReadVariableOp Adam/lstm_27/lstm_cell_27/bias/m*
_output_shapes	
:*
dtype0
Е
,Adam/lstm_27/lstm_cell_27/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	d*=
shared_name.,Adam/lstm_27/lstm_cell_27/recurrent_kernel/m
Ў
@Adam/lstm_27/lstm_cell_27/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp,Adam/lstm_27/lstm_cell_27/recurrent_kernel/m*
_output_shapes
:	d*
dtype0
Ё
"Adam/lstm_27/lstm_cell_27/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*3
shared_name$"Adam/lstm_27/lstm_cell_27/kernel/m

6Adam/lstm_27/lstm_cell_27/kernel/m/Read/ReadVariableOpReadVariableOp"Adam/lstm_27/lstm_cell_27/kernel/m*
_output_shapes
:	*
dtype0

Adam/dense_83/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_83/bias/m
y
(Adam/dense_83/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_83/bias/m*
_output_shapes
:*
dtype0

Adam/dense_83/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*'
shared_nameAdam/dense_83/kernel/m

*Adam/dense_83/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_83/kernel/m*
_output_shapes

:2*
dtype0

Adam/dense_82/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*%
shared_nameAdam/dense_82/bias/m
y
(Adam/dense_82/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_82/bias/m*
_output_shapes
:2*
dtype0

Adam/dense_82/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:22*'
shared_nameAdam/dense_82/kernel/m

*Adam/dense_82/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_82/kernel/m*
_output_shapes

:22*
dtype0

Adam/dense_81/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*%
shared_nameAdam/dense_81/bias/m
y
(Adam/dense_81/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_81/bias/m*
_output_shapes
:2*
dtype0

Adam/dense_81/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d2*'
shared_nameAdam/dense_81/kernel/m

*Adam/dense_81/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_81/kernel/m*
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

lstm_27/lstm_cell_27/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_namelstm_27/lstm_cell_27/bias

-lstm_27/lstm_cell_27/bias/Read/ReadVariableOpReadVariableOplstm_27/lstm_cell_27/bias*
_output_shapes	
:*
dtype0
Ї
%lstm_27/lstm_cell_27/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	d*6
shared_name'%lstm_27/lstm_cell_27/recurrent_kernel
 
9lstm_27/lstm_cell_27/recurrent_kernel/Read/ReadVariableOpReadVariableOp%lstm_27/lstm_cell_27/recurrent_kernel*
_output_shapes
:	d*
dtype0

lstm_27/lstm_cell_27/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*,
shared_namelstm_27/lstm_cell_27/kernel

/lstm_27/lstm_cell_27/kernel/Read/ReadVariableOpReadVariableOplstm_27/lstm_cell_27/kernel*
_output_shapes
:	*
dtype0
r
dense_83/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_83/bias
k
!dense_83/bias/Read/ReadVariableOpReadVariableOpdense_83/bias*
_output_shapes
:*
dtype0
z
dense_83/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2* 
shared_namedense_83/kernel
s
#dense_83/kernel/Read/ReadVariableOpReadVariableOpdense_83/kernel*
_output_shapes

:2*
dtype0
r
dense_82/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*
shared_namedense_82/bias
k
!dense_82/bias/Read/ReadVariableOpReadVariableOpdense_82/bias*
_output_shapes
:2*
dtype0
z
dense_82/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:22* 
shared_namedense_82/kernel
s
#dense_82/kernel/Read/ReadVariableOpReadVariableOpdense_82/kernel*
_output_shapes

:22*
dtype0
r
dense_81/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*
shared_namedense_81/bias
k
!dense_81/bias/Read/ReadVariableOpReadVariableOpdense_81/bias*
_output_shapes
:2*
dtype0
z
dense_81/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d2* 
shared_namedense_81/kernel
s
#dense_81/kernel/Read/ReadVariableOpReadVariableOpdense_81/kernel*
_output_shapes

:d2*
dtype0

serving_default_lstm_27_inputPlaceholder*+
_output_shapes
:џџџџџџџџџ*
dtype0* 
shape:џџџџџџџџџ

StatefulPartitionedCallStatefulPartitionedCallserving_default_lstm_27_inputlstm_27/lstm_cell_27/kernel%lstm_27/lstm_cell_27/recurrent_kernellstm_27/lstm_cell_27/biasdense_81/kerneldense_81/biasdense_82/kerneldense_82/biasdense_83/kerneldense_83/bias*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*+
_read_only_resource_inputs
		*-
config_proto

CPU

GPU 2J 8 *-
f(R&
$__inference_signature_wrapper_940770

NoOpNoOp
ЪC
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*C
valueћBBјB BёB
ш
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
С
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
І
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias*
І
	variables
 trainable_variables
!regularization_losses
"	keras_api
#__call__
*$&call_and_return_all_conditional_losses

%kernel
&bias*
І
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
А
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
ј
?iter

@beta_1

Abeta_2
	Bdecay
Clearning_ratemm%m&m-m.m/m0m1mvv%v&v-v.v/v0v1v*
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


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
у
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

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
_Y
VARIABLE_VALUEdense_81/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_81/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

%0
&1*

%0
&1*
* 

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
_Y
VARIABLE_VALUEdense_82/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_82/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

-0
.1*

-0
.1*
* 

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
_Y
VARIABLE_VALUEdense_83/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_83/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUElstm_27/lstm_cell_27/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUE%lstm_27/lstm_cell_27/recurrent_kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUElstm_27/lstm_cell_27/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
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

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
	keras_api

total

count

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
0
1*

	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
|
VARIABLE_VALUEAdam/dense_81/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_81/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_82/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_82/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_83/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_83/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE"Adam/lstm_27/lstm_cell_27/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE,Adam/lstm_27/lstm_cell_27/recurrent_kernel/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE Adam/lstm_27/lstm_cell_27/bias/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_81/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_81/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_82/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_82/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_83/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_83/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE"Adam/lstm_27/lstm_cell_27/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE,Adam/lstm_27/lstm_cell_27/recurrent_kernel/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE Adam/lstm_27/lstm_cell_27/bias/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Џ
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_81/kernel/Read/ReadVariableOp!dense_81/bias/Read/ReadVariableOp#dense_82/kernel/Read/ReadVariableOp!dense_82/bias/Read/ReadVariableOp#dense_83/kernel/Read/ReadVariableOp!dense_83/bias/Read/ReadVariableOp/lstm_27/lstm_cell_27/kernel/Read/ReadVariableOp9lstm_27/lstm_cell_27/recurrent_kernel/Read/ReadVariableOp-lstm_27/lstm_cell_27/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp*Adam/dense_81/kernel/m/Read/ReadVariableOp(Adam/dense_81/bias/m/Read/ReadVariableOp*Adam/dense_82/kernel/m/Read/ReadVariableOp(Adam/dense_82/bias/m/Read/ReadVariableOp*Adam/dense_83/kernel/m/Read/ReadVariableOp(Adam/dense_83/bias/m/Read/ReadVariableOp6Adam/lstm_27/lstm_cell_27/kernel/m/Read/ReadVariableOp@Adam/lstm_27/lstm_cell_27/recurrent_kernel/m/Read/ReadVariableOp4Adam/lstm_27/lstm_cell_27/bias/m/Read/ReadVariableOp*Adam/dense_81/kernel/v/Read/ReadVariableOp(Adam/dense_81/bias/v/Read/ReadVariableOp*Adam/dense_82/kernel/v/Read/ReadVariableOp(Adam/dense_82/bias/v/Read/ReadVariableOp*Adam/dense_83/kernel/v/Read/ReadVariableOp(Adam/dense_83/bias/v/Read/ReadVariableOp6Adam/lstm_27/lstm_cell_27/kernel/v/Read/ReadVariableOp@Adam/lstm_27/lstm_cell_27/recurrent_kernel/v/Read/ReadVariableOp4Adam/lstm_27/lstm_cell_27/bias/v/Read/ReadVariableOpConst*1
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
GPU 2J 8 *(
f#R!
__inference__traced_save_942058
к
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_81/kerneldense_81/biasdense_82/kerneldense_82/biasdense_83/kerneldense_83/biaslstm_27/lstm_cell_27/kernel%lstm_27/lstm_cell_27/recurrent_kernellstm_27/lstm_cell_27/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotal_1count_1totalcountAdam/dense_81/kernel/mAdam/dense_81/bias/mAdam/dense_82/kernel/mAdam/dense_82/bias/mAdam/dense_83/kernel/mAdam/dense_83/bias/m"Adam/lstm_27/lstm_cell_27/kernel/m,Adam/lstm_27/lstm_cell_27/recurrent_kernel/m Adam/lstm_27/lstm_cell_27/bias/mAdam/dense_81/kernel/vAdam/dense_81/bias/vAdam/dense_82/kernel/vAdam/dense_82/bias/vAdam/dense_83/kernel/vAdam/dense_83/bias/v"Adam/lstm_27/lstm_cell_27/kernel/v,Adam/lstm_27/lstm_cell_27/recurrent_kernel/v Adam/lstm_27/lstm_cell_27/bias/v*0
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
GPU 2J 8 *+
f&R$
"__inference__traced_restore_942176Те

З
(__inference_lstm_27_layer_call_fn_941157
inputs_0
unknown:	
	unknown_0:	d
	unknown_1:	
identityЂStatefulPartitionedCallч
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџd*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_lstm_27_layer_call_and_return_conditional_losses_939962o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџd`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџџџџџџџџџџ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/0
ЉK

C__inference_lstm_27_layer_call_and_return_conditional_losses_940315

inputs>
+lstm_cell_27_matmul_readvariableop_resource:	@
-lstm_cell_27_matmul_1_readvariableop_resource:	d;
,lstm_cell_27_biasadd_readvariableop_resource:	
identityЂ#lstm_cell_27/BiasAdd/ReadVariableOpЂ"lstm_cell_27/MatMul/ReadVariableOpЂ$lstm_cell_27/MatMul_1/ReadVariableOpЂwhile;
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
valueB:б
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
:џџџџџџџџџdR
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
:џџџџџџџџџdc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџD
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
valueB:л
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
џџџџџџџџџД
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв_
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
valueB:щ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask
"lstm_cell_27/MatMul/ReadVariableOpReadVariableOp+lstm_cell_27_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
lstm_cell_27/MatMulMatMulstrided_slice_2:output:0*lstm_cell_27/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ
$lstm_cell_27/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_27_matmul_1_readvariableop_resource*
_output_shapes
:	d*
dtype0
lstm_cell_27/MatMul_1MatMulzeros:output:0,lstm_cell_27/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ
lstm_cell_27/addAddV2lstm_cell_27/MatMul:product:0lstm_cell_27/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџ
#lstm_cell_27/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_27_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
lstm_cell_27/BiasAddBiasAddlstm_cell_27/add:z:0+lstm_cell_27/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ^
lstm_cell_27/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :н
lstm_cell_27/splitSplit%lstm_cell_27/split/split_dim:output:0lstm_cell_27/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџd:џџџџџџџџџd:џџџџџџџџџd:џџџџџџџџџd*
	num_splitn
lstm_cell_27/SigmoidSigmoidlstm_cell_27/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџdp
lstm_cell_27/Sigmoid_1Sigmoidlstm_cell_27/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџdw
lstm_cell_27/mulMullstm_cell_27/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџdh
lstm_cell_27/ReluRelulstm_cell_27/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџd
lstm_cell_27/mul_1Mullstm_cell_27/Sigmoid:y:0lstm_cell_27/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџd{
lstm_cell_27/add_1AddV2lstm_cell_27/mul:z:0lstm_cell_27/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџdp
lstm_cell_27/Sigmoid_2Sigmoidlstm_cell_27/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџde
lstm_cell_27/Relu_1Relulstm_cell_27/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџd
lstm_cell_27/mul_2Mullstm_cell_27/Sigmoid_2:y:0!lstm_cell_27/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџdn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџd   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Х
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвF
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
џџџџџџџџџT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_27_matmul_readvariableop_resource-lstm_cell_27_matmul_1_readvariableop_resource,lstm_cell_27_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџd:џџџџџџџџџd: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_940230*
condR
while_cond_940229*K
output_shapes:
8: : : : :џџџџџџџџџd:џџџџџџџџџd: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџd   ж
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџd*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџd*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџd[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџdР
NoOpNoOp$^lstm_cell_27/BiasAdd/ReadVariableOp#^lstm_cell_27/MatMul/ReadVariableOp%^lstm_cell_27/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:џџџџџџџџџ: : : 2J
#lstm_cell_27/BiasAdd/ReadVariableOp#lstm_cell_27/BiasAdd/ReadVariableOp2H
"lstm_cell_27/MatMul/ReadVariableOp"lstm_cell_27/MatMul/ReadVariableOp2L
$lstm_cell_27/MatMul_1/ReadVariableOp$lstm_cell_27/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
И
Ѕ
I__inference_sequential_27_layer_call_and_return_conditional_losses_940713
lstm_27_input!
lstm_27_940690:	!
lstm_27_940692:	d
lstm_27_940694:	!
dense_81_940697:d2
dense_81_940699:2!
dense_82_940702:22
dense_82_940704:2!
dense_83_940707:2
dense_83_940709:
identityЂ dense_81/StatefulPartitionedCallЂ dense_82/StatefulPartitionedCallЂ dense_83/StatefulPartitionedCallЂlstm_27/StatefulPartitionedCall
lstm_27/StatefulPartitionedCallStatefulPartitionedCalllstm_27_inputlstm_27_940690lstm_27_940692lstm_27_940694*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџd*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_lstm_27_layer_call_and_return_conditional_losses_940315
 dense_81/StatefulPartitionedCallStatefulPartitionedCall(lstm_27/StatefulPartitionedCall:output:0dense_81_940697dense_81_940699*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_81_layer_call_and_return_conditional_losses_940334
 dense_82/StatefulPartitionedCallStatefulPartitionedCall)dense_81/StatefulPartitionedCall:output:0dense_82_940702dense_82_940704*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_82_layer_call_and_return_conditional_losses_940351
 dense_83/StatefulPartitionedCallStatefulPartitionedCall)dense_82/StatefulPartitionedCall:output:0dense_83_940707dense_83_940709*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_83_layer_call_and_return_conditional_losses_940367x
IdentityIdentity)dense_83/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџб
NoOpNoOp!^dense_81/StatefulPartitionedCall!^dense_82/StatefulPartitionedCall!^dense_83/StatefulPartitionedCall ^lstm_27/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):џџџџџџџџџ: : : : : : : : : 2D
 dense_81/StatefulPartitionedCall dense_81/StatefulPartitionedCall2D
 dense_82/StatefulPartitionedCall dense_82/StatefulPartitionedCall2D
 dense_83/StatefulPartitionedCall dense_83/StatefulPartitionedCall2B
lstm_27/StatefulPartitionedCalllstm_27/StatefulPartitionedCall:Z V
+
_output_shapes
:џџџџџџџџџ
'
_user_specified_namelstm_27_input
Ч	
ѕ
D__inference_dense_83_layer_call_and_return_conditional_losses_940367

inputs0
matmul_readvariableop_resource:2-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:2*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ2: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ2
 
_user_specified_nameinputs


ѕ
D__inference_dense_81_layer_call_and_return_conditional_losses_940334

inputs0
matmul_readvariableop_resource:d2-
biasadd_readvariableop_resource:2
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d2*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџd: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџd
 
_user_specified_nameinputs
ЉK

C__inference_lstm_27_layer_call_and_return_conditional_losses_940583

inputs>
+lstm_cell_27_matmul_readvariableop_resource:	@
-lstm_cell_27_matmul_1_readvariableop_resource:	d;
,lstm_cell_27_biasadd_readvariableop_resource:	
identityЂ#lstm_cell_27/BiasAdd/ReadVariableOpЂ"lstm_cell_27/MatMul/ReadVariableOpЂ$lstm_cell_27/MatMul_1/ReadVariableOpЂwhile;
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
valueB:б
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
:џџџџџџџџџdR
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
:џџџџџџџџџdc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџD
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
valueB:л
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
џџџџџџџџџД
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв_
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
valueB:щ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask
"lstm_cell_27/MatMul/ReadVariableOpReadVariableOp+lstm_cell_27_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
lstm_cell_27/MatMulMatMulstrided_slice_2:output:0*lstm_cell_27/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ
$lstm_cell_27/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_27_matmul_1_readvariableop_resource*
_output_shapes
:	d*
dtype0
lstm_cell_27/MatMul_1MatMulzeros:output:0,lstm_cell_27/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ
lstm_cell_27/addAddV2lstm_cell_27/MatMul:product:0lstm_cell_27/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџ
#lstm_cell_27/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_27_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
lstm_cell_27/BiasAddBiasAddlstm_cell_27/add:z:0+lstm_cell_27/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ^
lstm_cell_27/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :н
lstm_cell_27/splitSplit%lstm_cell_27/split/split_dim:output:0lstm_cell_27/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџd:џџџџџџџџџd:џџџџџџџџџd:џџџџџџџџџd*
	num_splitn
lstm_cell_27/SigmoidSigmoidlstm_cell_27/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџdp
lstm_cell_27/Sigmoid_1Sigmoidlstm_cell_27/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџdw
lstm_cell_27/mulMullstm_cell_27/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџdh
lstm_cell_27/ReluRelulstm_cell_27/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџd
lstm_cell_27/mul_1Mullstm_cell_27/Sigmoid:y:0lstm_cell_27/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџd{
lstm_cell_27/add_1AddV2lstm_cell_27/mul:z:0lstm_cell_27/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџdp
lstm_cell_27/Sigmoid_2Sigmoidlstm_cell_27/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџde
lstm_cell_27/Relu_1Relulstm_cell_27/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџd
lstm_cell_27/mul_2Mullstm_cell_27/Sigmoid_2:y:0!lstm_cell_27/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџdn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџd   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Х
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвF
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
џџџџџџџџџT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_27_matmul_readvariableop_resource-lstm_cell_27_matmul_1_readvariableop_resource,lstm_cell_27_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџd:џџџџџџџџџd: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_940498*
condR
while_cond_940497*K
output_shapes:
8: : : : :џџџџџџџџџd:џџџџџџџџџd: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџd   ж
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџd*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџd*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџd[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџdР
NoOpNoOp$^lstm_cell_27/BiasAdd/ReadVariableOp#^lstm_cell_27/MatMul/ReadVariableOp%^lstm_cell_27/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:џџџџџџџџџ: : : 2J
#lstm_cell_27/BiasAdd/ReadVariableOp#lstm_cell_27/BiasAdd/ReadVariableOp2H
"lstm_cell_27/MatMul/ReadVariableOp"lstm_cell_27/MatMul/ReadVariableOp2L
$lstm_cell_27/MatMul_1/ReadVariableOp$lstm_cell_27/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ј9
а
while_body_941250
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_27_matmul_readvariableop_resource_0:	H
5while_lstm_cell_27_matmul_1_readvariableop_resource_0:	dC
4while_lstm_cell_27_biasadd_readvariableop_resource_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_27_matmul_readvariableop_resource:	F
3while_lstm_cell_27_matmul_1_readvariableop_resource:	dA
2while_lstm_cell_27_biasadd_readvariableop_resource:	Ђ)while/lstm_cell_27/BiasAdd/ReadVariableOpЂ(while/lstm_cell_27/MatMul/ReadVariableOpЂ*while/lstm_cell_27/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   І
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype0
(while/lstm_cell_27/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_27_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype0К
while/lstm_cell_27/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_27/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЁ
*while/lstm_cell_27/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_27_matmul_1_readvariableop_resource_0*
_output_shapes
:	d*
dtype0Ё
while/lstm_cell_27/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_27/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ
while/lstm_cell_27/addAddV2#while/lstm_cell_27/MatMul:product:0%while/lstm_cell_27/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџ
)while/lstm_cell_27/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_27_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0Ї
while/lstm_cell_27/BiasAddBiasAddwhile/lstm_cell_27/add:z:01while/lstm_cell_27/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџd
"while/lstm_cell_27/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :я
while/lstm_cell_27/splitSplit+while/lstm_cell_27/split/split_dim:output:0#while/lstm_cell_27/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџd:џџџџџџџџџd:џџџџџџџџџd:џџџџџџџџџd*
	num_splitz
while/lstm_cell_27/SigmoidSigmoid!while/lstm_cell_27/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџd|
while/lstm_cell_27/Sigmoid_1Sigmoid!while/lstm_cell_27/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџd
while/lstm_cell_27/mulMul while/lstm_cell_27/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџdt
while/lstm_cell_27/ReluRelu!while/lstm_cell_27/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџd
while/lstm_cell_27/mul_1Mulwhile/lstm_cell_27/Sigmoid:y:0%while/lstm_cell_27/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџd
while/lstm_cell_27/add_1AddV2while/lstm_cell_27/mul:z:0while/lstm_cell_27/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџd|
while/lstm_cell_27/Sigmoid_2Sigmoid!while/lstm_cell_27/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџdq
while/lstm_cell_27/Relu_1Reluwhile/lstm_cell_27/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџd
while/lstm_cell_27/mul_2Mul while/lstm_cell_27/Sigmoid_2:y:0'while/lstm_cell_27/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџdr
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : э
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell_27/mul_2:z:0*
_output_shapes
: *
element_dtype0:щшвM
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
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: y
while/Identity_4Identitywhile/lstm_cell_27/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџdy
while/Identity_5Identitywhile/lstm_cell_27/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџdа

while/NoOpNoOp*^while/lstm_cell_27/BiasAdd/ReadVariableOp)^while/lstm_cell_27/MatMul/ReadVariableOp+^while/lstm_cell_27/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_27_biasadd_readvariableop_resource4while_lstm_cell_27_biasadd_readvariableop_resource_0"l
3while_lstm_cell_27_matmul_1_readvariableop_resource5while_lstm_cell_27_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_27_matmul_readvariableop_resource3while_lstm_cell_27_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџd:џџџџџџџџџd: : : : : 2V
)while/lstm_cell_27/BiasAdd/ReadVariableOp)while/lstm_cell_27/BiasAdd/ReadVariableOp2T
(while/lstm_cell_27/MatMul/ReadVariableOp(while/lstm_cell_27/MatMul/ReadVariableOp2X
*while/lstm_cell_27/MatMul_1/ReadVariableOp*while/lstm_cell_27/MatMul_1/ReadVariableOp: 
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
:џџџџџџџџџd:-)
'
_output_shapes
:џџџџџџџџџd:

_output_shapes
: :

_output_shapes
: 

ћ
'sequential_27_lstm_27_while_cond_939704H
Dsequential_27_lstm_27_while_sequential_27_lstm_27_while_loop_counterN
Jsequential_27_lstm_27_while_sequential_27_lstm_27_while_maximum_iterations+
'sequential_27_lstm_27_while_placeholder-
)sequential_27_lstm_27_while_placeholder_1-
)sequential_27_lstm_27_while_placeholder_2-
)sequential_27_lstm_27_while_placeholder_3J
Fsequential_27_lstm_27_while_less_sequential_27_lstm_27_strided_slice_1`
\sequential_27_lstm_27_while_sequential_27_lstm_27_while_cond_939704___redundant_placeholder0`
\sequential_27_lstm_27_while_sequential_27_lstm_27_while_cond_939704___redundant_placeholder1`
\sequential_27_lstm_27_while_sequential_27_lstm_27_while_cond_939704___redundant_placeholder2`
\sequential_27_lstm_27_while_sequential_27_lstm_27_while_cond_939704___redundant_placeholder3(
$sequential_27_lstm_27_while_identity
К
 sequential_27/lstm_27/while/LessLess'sequential_27_lstm_27_while_placeholderFsequential_27_lstm_27_while_less_sequential_27_lstm_27_strided_slice_1*
T0*
_output_shapes
: w
$sequential_27/lstm_27/while/IdentityIdentity$sequential_27/lstm_27/while/Less:z:0*
T0
*
_output_shapes
: "U
$sequential_27_lstm_27_while_identity-sequential_27/lstm_27/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :џџџџџџџџџd:џџџџџџџџџd: ::::: 
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
:џџџџџџџџџd:-)
'
_output_shapes
:џџџџџџџџџd:

_output_shapes
: :

_output_shapes
:
Е
У
while_cond_940497
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_940497___redundant_placeholder04
0while_while_cond_940497___redundant_placeholder14
0while_while_cond_940497___redundant_placeholder24
0while_while_cond_940497___redundant_placeholder3
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
@: : : : :џџџџџџџџџd:џџџџџџџџџd: ::::: 
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
:џџџџџџџџџd:-)
'
_output_shapes
:џџџџџџџџџd:

_output_shapes
: :

_output_shapes
:
ЬK

C__inference_lstm_27_layer_call_and_return_conditional_losses_941480
inputs_0>
+lstm_cell_27_matmul_readvariableop_resource:	@
-lstm_cell_27_matmul_1_readvariableop_resource:	d;
,lstm_cell_27_biasadd_readvariableop_resource:	
identityЂ#lstm_cell_27/BiasAdd/ReadVariableOpЂ"lstm_cell_27/MatMul/ReadVariableOpЂ$lstm_cell_27/MatMul_1/ReadVariableOpЂwhile=
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
valueB:б
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
:џџџџџџџџџdR
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
:џџџџџџџџџdc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџD
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
valueB:л
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
џџџџџџџџџД
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв_
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
valueB:щ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask
"lstm_cell_27/MatMul/ReadVariableOpReadVariableOp+lstm_cell_27_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
lstm_cell_27/MatMulMatMulstrided_slice_2:output:0*lstm_cell_27/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ
$lstm_cell_27/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_27_matmul_1_readvariableop_resource*
_output_shapes
:	d*
dtype0
lstm_cell_27/MatMul_1MatMulzeros:output:0,lstm_cell_27/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ
lstm_cell_27/addAddV2lstm_cell_27/MatMul:product:0lstm_cell_27/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџ
#lstm_cell_27/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_27_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
lstm_cell_27/BiasAddBiasAddlstm_cell_27/add:z:0+lstm_cell_27/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ^
lstm_cell_27/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :н
lstm_cell_27/splitSplit%lstm_cell_27/split/split_dim:output:0lstm_cell_27/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџd:џџџџџџџџџd:џџџџџџџџџd:џџџџџџџџџd*
	num_splitn
lstm_cell_27/SigmoidSigmoidlstm_cell_27/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџdp
lstm_cell_27/Sigmoid_1Sigmoidlstm_cell_27/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџdw
lstm_cell_27/mulMullstm_cell_27/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџdh
lstm_cell_27/ReluRelulstm_cell_27/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџd
lstm_cell_27/mul_1Mullstm_cell_27/Sigmoid:y:0lstm_cell_27/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџd{
lstm_cell_27/add_1AddV2lstm_cell_27/mul:z:0lstm_cell_27/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџdp
lstm_cell_27/Sigmoid_2Sigmoidlstm_cell_27/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџde
lstm_cell_27/Relu_1Relulstm_cell_27/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџd
lstm_cell_27/mul_2Mullstm_cell_27/Sigmoid_2:y:0!lstm_cell_27/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџdn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџd   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Х
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвF
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
џџџџџџџџџT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_27_matmul_readvariableop_resource-lstm_cell_27_matmul_1_readvariableop_resource,lstm_cell_27_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџd:џџџџџџџџџd: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_941395*
condR
while_cond_941394*K
output_shapes:
8: : : : :џџџџџџџџџd:џџџџџџџџџd: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџd   ж
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџd*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџd*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџd[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџdР
NoOpNoOp$^lstm_cell_27/BiasAdd/ReadVariableOp#^lstm_cell_27/MatMul/ReadVariableOp%^lstm_cell_27/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџџџџџџџџџџ: : : 2J
#lstm_cell_27/BiasAdd/ReadVariableOp#lstm_cell_27/BiasAdd/ReadVariableOp2H
"lstm_cell_27/MatMul/ReadVariableOp"lstm_cell_27/MatMul/ReadVariableOp2L
$lstm_cell_27/MatMul_1/ReadVariableOp$lstm_cell_27/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/0
ЉK

C__inference_lstm_27_layer_call_and_return_conditional_losses_941770

inputs>
+lstm_cell_27_matmul_readvariableop_resource:	@
-lstm_cell_27_matmul_1_readvariableop_resource:	d;
,lstm_cell_27_biasadd_readvariableop_resource:	
identityЂ#lstm_cell_27/BiasAdd/ReadVariableOpЂ"lstm_cell_27/MatMul/ReadVariableOpЂ$lstm_cell_27/MatMul_1/ReadVariableOpЂwhile;
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
valueB:б
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
:џџџџџџџџџdR
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
:џџџџџџџџџdc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџD
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
valueB:л
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
џџџџџџџџџД
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв_
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
valueB:щ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask
"lstm_cell_27/MatMul/ReadVariableOpReadVariableOp+lstm_cell_27_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
lstm_cell_27/MatMulMatMulstrided_slice_2:output:0*lstm_cell_27/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ
$lstm_cell_27/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_27_matmul_1_readvariableop_resource*
_output_shapes
:	d*
dtype0
lstm_cell_27/MatMul_1MatMulzeros:output:0,lstm_cell_27/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ
lstm_cell_27/addAddV2lstm_cell_27/MatMul:product:0lstm_cell_27/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџ
#lstm_cell_27/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_27_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
lstm_cell_27/BiasAddBiasAddlstm_cell_27/add:z:0+lstm_cell_27/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ^
lstm_cell_27/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :н
lstm_cell_27/splitSplit%lstm_cell_27/split/split_dim:output:0lstm_cell_27/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџd:џџџџџџџџџd:џџџџџџџџџd:џџџџџџџџџd*
	num_splitn
lstm_cell_27/SigmoidSigmoidlstm_cell_27/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџdp
lstm_cell_27/Sigmoid_1Sigmoidlstm_cell_27/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџdw
lstm_cell_27/mulMullstm_cell_27/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџdh
lstm_cell_27/ReluRelulstm_cell_27/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџd
lstm_cell_27/mul_1Mullstm_cell_27/Sigmoid:y:0lstm_cell_27/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџd{
lstm_cell_27/add_1AddV2lstm_cell_27/mul:z:0lstm_cell_27/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџdp
lstm_cell_27/Sigmoid_2Sigmoidlstm_cell_27/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџde
lstm_cell_27/Relu_1Relulstm_cell_27/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџd
lstm_cell_27/mul_2Mullstm_cell_27/Sigmoid_2:y:0!lstm_cell_27/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџdn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџd   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Х
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвF
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
џџџџџџџџџT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_27_matmul_readvariableop_resource-lstm_cell_27_matmul_1_readvariableop_resource,lstm_cell_27_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџd:џџџџџџџџџd: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_941685*
condR
while_cond_941684*K
output_shapes:
8: : : : :џџџџџџџџџd:џџџџџџџџџd: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџd   ж
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџd*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџd*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџd[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџdР
NoOpNoOp$^lstm_cell_27/BiasAdd/ReadVariableOp#^lstm_cell_27/MatMul/ReadVariableOp%^lstm_cell_27/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:џџџџџџџџџ: : : 2J
#lstm_cell_27/BiasAdd/ReadVariableOp#lstm_cell_27/BiasAdd/ReadVariableOp2H
"lstm_cell_27/MatMul/ReadVariableOp"lstm_cell_27/MatMul/ReadVariableOp2L
$lstm_cell_27/MatMul_1/ReadVariableOp$lstm_cell_27/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs


у
lstm_27_while_cond_940875,
(lstm_27_while_lstm_27_while_loop_counter2
.lstm_27_while_lstm_27_while_maximum_iterations
lstm_27_while_placeholder
lstm_27_while_placeholder_1
lstm_27_while_placeholder_2
lstm_27_while_placeholder_3.
*lstm_27_while_less_lstm_27_strided_slice_1D
@lstm_27_while_lstm_27_while_cond_940875___redundant_placeholder0D
@lstm_27_while_lstm_27_while_cond_940875___redundant_placeholder1D
@lstm_27_while_lstm_27_while_cond_940875___redundant_placeholder2D
@lstm_27_while_lstm_27_while_cond_940875___redundant_placeholder3
lstm_27_while_identity

lstm_27/while/LessLesslstm_27_while_placeholder*lstm_27_while_less_lstm_27_strided_slice_1*
T0*
_output_shapes
: [
lstm_27/while/IdentityIdentitylstm_27/while/Less:z:0*
T0
*
_output_shapes
: "9
lstm_27_while_identitylstm_27/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :џџџџџџџџџd:џџџџџџџџџd: ::::: 
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
:џџџџџџџџџd:-)
'
_output_shapes
:џџџџџџџџџd:

_output_shapes
: :

_output_shapes
:
Т

)__inference_dense_81_layer_call_fn_941779

inputs
unknown:d2
	unknown_0:2
identityЂStatefulPartitionedCallй
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_81_layer_call_and_return_conditional_losses_940334o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџd: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџd
 
_user_specified_nameinputs
§8

C__inference_lstm_27_layer_call_and_return_conditional_losses_940155

inputs&
lstm_cell_27_940071:	&
lstm_cell_27_940073:	d"
lstm_cell_27_940075:	
identityЂ$lstm_cell_27/StatefulPartitionedCallЂwhile;
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
valueB:б
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
:џџџџџџџџџdR
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
:џџџџџџџџџdc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџD
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
valueB:л
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
џџџџџџџџџД
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв_
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
valueB:щ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_maskѕ
$lstm_cell_27/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_27_940071lstm_cell_27_940073lstm_cell_27_940075*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:џџџџџџџџџd:џџџџџџџџџd:џџџџџџџџџd*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_lstm_cell_27_layer_call_and_return_conditional_losses_940025n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџd   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Х
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвF
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
џџџџџџџџџT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : З
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_27_940071lstm_cell_27_940073lstm_cell_27_940075*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџd:џџџџџџџџџd: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_940085*
condR
while_cond_940084*K
output_shapes:
8: : : : :џџџџџџџџџd:џџџџџџџџџd: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџd   ж
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџd*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџd*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџd[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџdu
NoOpNoOp%^lstm_cell_27/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџџџџџџџџџџ: : : 2L
$lstm_cell_27/StatefulPartitionedCall$lstm_cell_27/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
щB
а

lstm_27_while_body_941041,
(lstm_27_while_lstm_27_while_loop_counter2
.lstm_27_while_lstm_27_while_maximum_iterations
lstm_27_while_placeholder
lstm_27_while_placeholder_1
lstm_27_while_placeholder_2
lstm_27_while_placeholder_3+
'lstm_27_while_lstm_27_strided_slice_1_0g
clstm_27_while_tensorarrayv2read_tensorlistgetitem_lstm_27_tensorarrayunstack_tensorlistfromtensor_0N
;lstm_27_while_lstm_cell_27_matmul_readvariableop_resource_0:	P
=lstm_27_while_lstm_cell_27_matmul_1_readvariableop_resource_0:	dK
<lstm_27_while_lstm_cell_27_biasadd_readvariableop_resource_0:	
lstm_27_while_identity
lstm_27_while_identity_1
lstm_27_while_identity_2
lstm_27_while_identity_3
lstm_27_while_identity_4
lstm_27_while_identity_5)
%lstm_27_while_lstm_27_strided_slice_1e
alstm_27_while_tensorarrayv2read_tensorlistgetitem_lstm_27_tensorarrayunstack_tensorlistfromtensorL
9lstm_27_while_lstm_cell_27_matmul_readvariableop_resource:	N
;lstm_27_while_lstm_cell_27_matmul_1_readvariableop_resource:	dI
:lstm_27_while_lstm_cell_27_biasadd_readvariableop_resource:	Ђ1lstm_27/while/lstm_cell_27/BiasAdd/ReadVariableOpЂ0lstm_27/while/lstm_cell_27/MatMul/ReadVariableOpЂ2lstm_27/while/lstm_cell_27/MatMul_1/ReadVariableOp
?lstm_27/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   Ю
1lstm_27/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_27_while_tensorarrayv2read_tensorlistgetitem_lstm_27_tensorarrayunstack_tensorlistfromtensor_0lstm_27_while_placeholderHlstm_27/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype0­
0lstm_27/while/lstm_cell_27/MatMul/ReadVariableOpReadVariableOp;lstm_27_while_lstm_cell_27_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype0в
!lstm_27/while/lstm_cell_27/MatMulMatMul8lstm_27/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_27/while/lstm_cell_27/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџБ
2lstm_27/while/lstm_cell_27/MatMul_1/ReadVariableOpReadVariableOp=lstm_27_while_lstm_cell_27_matmul_1_readvariableop_resource_0*
_output_shapes
:	d*
dtype0Й
#lstm_27/while/lstm_cell_27/MatMul_1MatMullstm_27_while_placeholder_2:lstm_27/while/lstm_cell_27/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЖ
lstm_27/while/lstm_cell_27/addAddV2+lstm_27/while/lstm_cell_27/MatMul:product:0-lstm_27/while/lstm_cell_27/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџЋ
1lstm_27/while/lstm_cell_27/BiasAdd/ReadVariableOpReadVariableOp<lstm_27_while_lstm_cell_27_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0П
"lstm_27/while/lstm_cell_27/BiasAddBiasAdd"lstm_27/while/lstm_cell_27/add:z:09lstm_27/while/lstm_cell_27/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџl
*lstm_27/while/lstm_cell_27/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
 lstm_27/while/lstm_cell_27/splitSplit3lstm_27/while/lstm_cell_27/split/split_dim:output:0+lstm_27/while/lstm_cell_27/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџd:џџџџџџџџџd:џџџџџџџџџd:џџџџџџџџџd*
	num_split
"lstm_27/while/lstm_cell_27/SigmoidSigmoid)lstm_27/while/lstm_cell_27/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџd
$lstm_27/while/lstm_cell_27/Sigmoid_1Sigmoid)lstm_27/while/lstm_cell_27/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџd
lstm_27/while/lstm_cell_27/mulMul(lstm_27/while/lstm_cell_27/Sigmoid_1:y:0lstm_27_while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџd
lstm_27/while/lstm_cell_27/ReluRelu)lstm_27/while/lstm_cell_27/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџdА
 lstm_27/while/lstm_cell_27/mul_1Mul&lstm_27/while/lstm_cell_27/Sigmoid:y:0-lstm_27/while/lstm_cell_27/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџdЅ
 lstm_27/while/lstm_cell_27/add_1AddV2"lstm_27/while/lstm_cell_27/mul:z:0$lstm_27/while/lstm_cell_27/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџd
$lstm_27/while/lstm_cell_27/Sigmoid_2Sigmoid)lstm_27/while/lstm_cell_27/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџd
!lstm_27/while/lstm_cell_27/Relu_1Relu$lstm_27/while/lstm_cell_27/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџdД
 lstm_27/while/lstm_cell_27/mul_2Mul(lstm_27/while/lstm_cell_27/Sigmoid_2:y:0/lstm_27/while/lstm_cell_27/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџdz
8lstm_27/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : 
2lstm_27/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_27_while_placeholder_1Alstm_27/while/TensorArrayV2Write/TensorListSetItem/index:output:0$lstm_27/while/lstm_cell_27/mul_2:z:0*
_output_shapes
: *
element_dtype0:щшвU
lstm_27/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :t
lstm_27/while/addAddV2lstm_27_while_placeholderlstm_27/while/add/y:output:0*
T0*
_output_shapes
: W
lstm_27/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
lstm_27/while/add_1AddV2(lstm_27_while_lstm_27_while_loop_counterlstm_27/while/add_1/y:output:0*
T0*
_output_shapes
: q
lstm_27/while/IdentityIdentitylstm_27/while/add_1:z:0^lstm_27/while/NoOp*
T0*
_output_shapes
: 
lstm_27/while/Identity_1Identity.lstm_27_while_lstm_27_while_maximum_iterations^lstm_27/while/NoOp*
T0*
_output_shapes
: q
lstm_27/while/Identity_2Identitylstm_27/while/add:z:0^lstm_27/while/NoOp*
T0*
_output_shapes
: 
lstm_27/while/Identity_3IdentityBlstm_27/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_27/while/NoOp*
T0*
_output_shapes
: 
lstm_27/while/Identity_4Identity$lstm_27/while/lstm_cell_27/mul_2:z:0^lstm_27/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџd
lstm_27/while/Identity_5Identity$lstm_27/while/lstm_cell_27/add_1:z:0^lstm_27/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџd№
lstm_27/while/NoOpNoOp2^lstm_27/while/lstm_cell_27/BiasAdd/ReadVariableOp1^lstm_27/while/lstm_cell_27/MatMul/ReadVariableOp3^lstm_27/while/lstm_cell_27/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "9
lstm_27_while_identitylstm_27/while/Identity:output:0"=
lstm_27_while_identity_1!lstm_27/while/Identity_1:output:0"=
lstm_27_while_identity_2!lstm_27/while/Identity_2:output:0"=
lstm_27_while_identity_3!lstm_27/while/Identity_3:output:0"=
lstm_27_while_identity_4!lstm_27/while/Identity_4:output:0"=
lstm_27_while_identity_5!lstm_27/while/Identity_5:output:0"P
%lstm_27_while_lstm_27_strided_slice_1'lstm_27_while_lstm_27_strided_slice_1_0"z
:lstm_27_while_lstm_cell_27_biasadd_readvariableop_resource<lstm_27_while_lstm_cell_27_biasadd_readvariableop_resource_0"|
;lstm_27_while_lstm_cell_27_matmul_1_readvariableop_resource=lstm_27_while_lstm_cell_27_matmul_1_readvariableop_resource_0"x
9lstm_27_while_lstm_cell_27_matmul_readvariableop_resource;lstm_27_while_lstm_cell_27_matmul_readvariableop_resource_0"Ш
alstm_27_while_tensorarrayv2read_tensorlistgetitem_lstm_27_tensorarrayunstack_tensorlistfromtensorclstm_27_while_tensorarrayv2read_tensorlistgetitem_lstm_27_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџd:џџџџџџџџџd: : : : : 2f
1lstm_27/while/lstm_cell_27/BiasAdd/ReadVariableOp1lstm_27/while/lstm_cell_27/BiasAdd/ReadVariableOp2d
0lstm_27/while/lstm_cell_27/MatMul/ReadVariableOp0lstm_27/while/lstm_cell_27/MatMul/ReadVariableOp2h
2lstm_27/while/lstm_cell_27/MatMul_1/ReadVariableOp2lstm_27/while/lstm_cell_27/MatMul_1/ReadVariableOp: 
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
:џџџџџџџџџd:-)
'
_output_shapes
:џџџџџџџџџd:

_output_shapes
: :

_output_shapes
: 


у
lstm_27_while_cond_941040,
(lstm_27_while_lstm_27_while_loop_counter2
.lstm_27_while_lstm_27_while_maximum_iterations
lstm_27_while_placeholder
lstm_27_while_placeholder_1
lstm_27_while_placeholder_2
lstm_27_while_placeholder_3.
*lstm_27_while_less_lstm_27_strided_slice_1D
@lstm_27_while_lstm_27_while_cond_941040___redundant_placeholder0D
@lstm_27_while_lstm_27_while_cond_941040___redundant_placeholder1D
@lstm_27_while_lstm_27_while_cond_941040___redundant_placeholder2D
@lstm_27_while_lstm_27_while_cond_941040___redundant_placeholder3
lstm_27_while_identity

lstm_27/while/LessLesslstm_27_while_placeholder*lstm_27_while_less_lstm_27_strided_slice_1*
T0*
_output_shapes
: [
lstm_27/while/IdentityIdentitylstm_27/while/Less:z:0*
T0
*
_output_shapes
: "9
lstm_27_while_identitylstm_27/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :џџџџџџџџџd:џџџџџџџџџd: ::::: 
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
:џџџџџџџџџd:-)
'
_output_shapes
:џџџџџџџџџd:

_output_shapes
: :

_output_shapes
:
Ѓ

I__inference_sequential_27_layer_call_and_return_conditional_losses_940374

inputs!
lstm_27_940316:	!
lstm_27_940318:	d
lstm_27_940320:	!
dense_81_940335:d2
dense_81_940337:2!
dense_82_940352:22
dense_82_940354:2!
dense_83_940368:2
dense_83_940370:
identityЂ dense_81/StatefulPartitionedCallЂ dense_82/StatefulPartitionedCallЂ dense_83/StatefulPartitionedCallЂlstm_27/StatefulPartitionedCallў
lstm_27/StatefulPartitionedCallStatefulPartitionedCallinputslstm_27_940316lstm_27_940318lstm_27_940320*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџd*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_lstm_27_layer_call_and_return_conditional_losses_940315
 dense_81/StatefulPartitionedCallStatefulPartitionedCall(lstm_27/StatefulPartitionedCall:output:0dense_81_940335dense_81_940337*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_81_layer_call_and_return_conditional_losses_940334
 dense_82/StatefulPartitionedCallStatefulPartitionedCall)dense_81/StatefulPartitionedCall:output:0dense_82_940352dense_82_940354*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_82_layer_call_and_return_conditional_losses_940351
 dense_83/StatefulPartitionedCallStatefulPartitionedCall)dense_82/StatefulPartitionedCall:output:0dense_83_940368dense_83_940370*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_83_layer_call_and_return_conditional_losses_940367x
IdentityIdentity)dense_83/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџб
NoOpNoOp!^dense_81/StatefulPartitionedCall!^dense_82/StatefulPartitionedCall!^dense_83/StatefulPartitionedCall ^lstm_27/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):џџџџџџџџџ: : : : : : : : : 2D
 dense_81/StatefulPartitionedCall dense_81/StatefulPartitionedCall2D
 dense_82/StatefulPartitionedCall dense_82/StatefulPartitionedCall2D
 dense_83/StatefulPartitionedCall dense_83/StatefulPartitionedCall2B
lstm_27/StatefulPartitionedCalllstm_27/StatefulPartitionedCall:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ы
і
-__inference_lstm_cell_27_layer_call_fn_941846

inputs
states_0
states_1
unknown:	
	unknown_0:	d
	unknown_1:	
identity

identity_1

identity_2ЂStatefulPartitionedCallЈ
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:џџџџџџџџџd:џџџџџџџџџd:џџџџџџџџџd*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_lstm_cell_27_layer_call_and_return_conditional_losses_939877o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџdq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:џџџџџџџџџdq

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:џџџџџџџџџd`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:џџџџџџџџџ:џџџџџџџџџd:џџџџџџџџџd: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:QM
'
_output_shapes
:џџџџџџџџџd
"
_user_specified_name
states/0:QM
'
_output_shapes
:џџџџџџџџџd
"
_user_specified_name
states/1


ф
.__inference_sequential_27_layer_call_fn_940687
lstm_27_input
unknown:	
	unknown_0:	d
	unknown_1:	
	unknown_2:d2
	unknown_3:2
	unknown_4:22
	unknown_5:2
	unknown_6:2
	unknown_7:
identityЂStatefulPartitionedCallР
StatefulPartitionedCallStatefulPartitionedCalllstm_27_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*+
_read_only_resource_inputs
		*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_27_layer_call_and_return_conditional_losses_940643o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):џџџџџџџџџ: : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
+
_output_shapes
:џџџџџџџџџ
'
_user_specified_namelstm_27_input
§8

C__inference_lstm_27_layer_call_and_return_conditional_losses_939962

inputs&
lstm_cell_27_939878:	&
lstm_cell_27_939880:	d"
lstm_cell_27_939882:	
identityЂ$lstm_cell_27/StatefulPartitionedCallЂwhile;
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
valueB:б
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
:џџџџџџџџџdR
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
:џџџџџџџџџdc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџD
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
valueB:л
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
џџџџџџџџџД
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв_
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
valueB:щ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_maskѕ
$lstm_cell_27/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_27_939878lstm_cell_27_939880lstm_cell_27_939882*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:џџџџџџџџџd:џџџџџџџџџd:џџџџџџџџџd*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_lstm_cell_27_layer_call_and_return_conditional_losses_939877n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџd   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Х
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвF
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
џџџџџџџџџT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : З
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_27_939878lstm_cell_27_939880lstm_cell_27_939882*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџd:џџџџџџџџџd: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_939892*
condR
while_cond_939891*K
output_shapes:
8: : : : :џџџџџџџџџd:џџџџџџџџџd: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџd   ж
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџd*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџd*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџd[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџdu
NoOpNoOp%^lstm_cell_27/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџџџџџџџџџџ: : : 2L
$lstm_cell_27/StatefulPartitionedCall$lstm_cell_27/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
m
ћ
I__inference_sequential_27_layer_call_and_return_conditional_losses_940981

inputsF
3lstm_27_lstm_cell_27_matmul_readvariableop_resource:	H
5lstm_27_lstm_cell_27_matmul_1_readvariableop_resource:	dC
4lstm_27_lstm_cell_27_biasadd_readvariableop_resource:	9
'dense_81_matmul_readvariableop_resource:d26
(dense_81_biasadd_readvariableop_resource:29
'dense_82_matmul_readvariableop_resource:226
(dense_82_biasadd_readvariableop_resource:29
'dense_83_matmul_readvariableop_resource:26
(dense_83_biasadd_readvariableop_resource:
identityЂdense_81/BiasAdd/ReadVariableOpЂdense_81/MatMul/ReadVariableOpЂdense_82/BiasAdd/ReadVariableOpЂdense_82/MatMul/ReadVariableOpЂdense_83/BiasAdd/ReadVariableOpЂdense_83/MatMul/ReadVariableOpЂ+lstm_27/lstm_cell_27/BiasAdd/ReadVariableOpЂ*lstm_27/lstm_cell_27/MatMul/ReadVariableOpЂ,lstm_27/lstm_cell_27/MatMul_1/ReadVariableOpЂlstm_27/whileC
lstm_27/ShapeShapeinputs*
T0*
_output_shapes
:e
lstm_27/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
lstm_27/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
lstm_27/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:љ
lstm_27/strided_sliceStridedSlicelstm_27/Shape:output:0$lstm_27/strided_slice/stack:output:0&lstm_27/strided_slice/stack_1:output:0&lstm_27/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
lstm_27/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d
lstm_27/zeros/packedPacklstm_27/strided_slice:output:0lstm_27/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:X
lstm_27/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
lstm_27/zerosFilllstm_27/zeros/packed:output:0lstm_27/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџdZ
lstm_27/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d
lstm_27/zeros_1/packedPacklstm_27/strided_slice:output:0!lstm_27/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:Z
lstm_27/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
lstm_27/zeros_1Filllstm_27/zeros_1/packed:output:0lstm_27/zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџdk
lstm_27/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          }
lstm_27/transpose	Transposeinputslstm_27/transpose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџT
lstm_27/Shape_1Shapelstm_27/transpose:y:0*
T0*
_output_shapes
:g
lstm_27/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: i
lstm_27/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
lstm_27/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
lstm_27/strided_slice_1StridedSlicelstm_27/Shape_1:output:0&lstm_27/strided_slice_1/stack:output:0(lstm_27/strided_slice_1/stack_1:output:0(lstm_27/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskn
#lstm_27/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџЬ
lstm_27/TensorArrayV2TensorListReserve,lstm_27/TensorArrayV2/element_shape:output:0 lstm_27/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
=lstm_27/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   ј
/lstm_27/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_27/transpose:y:0Flstm_27/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвg
lstm_27/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: i
lstm_27/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
lstm_27/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
lstm_27/strided_slice_2StridedSlicelstm_27/transpose:y:0&lstm_27/strided_slice_2/stack:output:0(lstm_27/strided_slice_2/stack_1:output:0(lstm_27/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask
*lstm_27/lstm_cell_27/MatMul/ReadVariableOpReadVariableOp3lstm_27_lstm_cell_27_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0Ў
lstm_27/lstm_cell_27/MatMulMatMul lstm_27/strided_slice_2:output:02lstm_27/lstm_cell_27/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЃ
,lstm_27/lstm_cell_27/MatMul_1/ReadVariableOpReadVariableOp5lstm_27_lstm_cell_27_matmul_1_readvariableop_resource*
_output_shapes
:	d*
dtype0Ј
lstm_27/lstm_cell_27/MatMul_1MatMullstm_27/zeros:output:04lstm_27/lstm_cell_27/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЄ
lstm_27/lstm_cell_27/addAddV2%lstm_27/lstm_cell_27/MatMul:product:0'lstm_27/lstm_cell_27/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџ
+lstm_27/lstm_cell_27/BiasAdd/ReadVariableOpReadVariableOp4lstm_27_lstm_cell_27_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0­
lstm_27/lstm_cell_27/BiasAddBiasAddlstm_27/lstm_cell_27/add:z:03lstm_27/lstm_cell_27/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџf
$lstm_27/lstm_cell_27/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ѕ
lstm_27/lstm_cell_27/splitSplit-lstm_27/lstm_cell_27/split/split_dim:output:0%lstm_27/lstm_cell_27/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџd:џџџџџџџџџd:џџџџџџџџџd:џџџџџџџџџd*
	num_split~
lstm_27/lstm_cell_27/SigmoidSigmoid#lstm_27/lstm_cell_27/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџd
lstm_27/lstm_cell_27/Sigmoid_1Sigmoid#lstm_27/lstm_cell_27/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџd
lstm_27/lstm_cell_27/mulMul"lstm_27/lstm_cell_27/Sigmoid_1:y:0lstm_27/zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџdx
lstm_27/lstm_cell_27/ReluRelu#lstm_27/lstm_cell_27/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџd
lstm_27/lstm_cell_27/mul_1Mul lstm_27/lstm_cell_27/Sigmoid:y:0'lstm_27/lstm_cell_27/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџd
lstm_27/lstm_cell_27/add_1AddV2lstm_27/lstm_cell_27/mul:z:0lstm_27/lstm_cell_27/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџd
lstm_27/lstm_cell_27/Sigmoid_2Sigmoid#lstm_27/lstm_cell_27/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџdu
lstm_27/lstm_cell_27/Relu_1Relulstm_27/lstm_cell_27/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџdЂ
lstm_27/lstm_cell_27/mul_2Mul"lstm_27/lstm_cell_27/Sigmoid_2:y:0)lstm_27/lstm_cell_27/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџdv
%lstm_27/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџd   f
$lstm_27/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :н
lstm_27/TensorArrayV2_1TensorListReserve.lstm_27/TensorArrayV2_1/element_shape:output:0-lstm_27/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвN
lstm_27/timeConst*
_output_shapes
: *
dtype0*
value	B : k
 lstm_27/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ\
lstm_27/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ђ
lstm_27/whileWhile#lstm_27/while/loop_counter:output:0)lstm_27/while/maximum_iterations:output:0lstm_27/time:output:0 lstm_27/TensorArrayV2_1:handle:0lstm_27/zeros:output:0lstm_27/zeros_1:output:0 lstm_27/strided_slice_1:output:0?lstm_27/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_27_lstm_cell_27_matmul_readvariableop_resource5lstm_27_lstm_cell_27_matmul_1_readvariableop_resource4lstm_27_lstm_cell_27_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџd:џџџџџџџџџd: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *%
bodyR
lstm_27_while_body_940876*%
condR
lstm_27_while_cond_940875*K
output_shapes:
8: : : : :џџџџџџџџџd:џџџџџџџџџd: : : : : *
parallel_iterations 
8lstm_27/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџd   ю
*lstm_27/TensorArrayV2Stack/TensorListStackTensorListStacklstm_27/while:output:3Alstm_27/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџd*
element_dtype0*
num_elementsp
lstm_27/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџi
lstm_27/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: i
lstm_27/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Џ
lstm_27/strided_slice_3StridedSlice3lstm_27/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_27/strided_slice_3/stack:output:0(lstm_27/strided_slice_3/stack_1:output:0(lstm_27/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџd*
shrink_axis_maskm
lstm_27/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ў
lstm_27/transpose_1	Transpose3lstm_27/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_27/transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџdc
lstm_27/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    
dense_81/MatMul/ReadVariableOpReadVariableOp'dense_81_matmul_readvariableop_resource*
_output_shapes

:d2*
dtype0
dense_81/MatMulMatMul lstm_27/strided_slice_3:output:0&dense_81/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_81/BiasAdd/ReadVariableOpReadVariableOp(dense_81_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0
dense_81/BiasAddBiasAdddense_81/MatMul:product:0'dense_81/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2b
dense_81/ReluReludense_81/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_82/MatMul/ReadVariableOpReadVariableOp'dense_82_matmul_readvariableop_resource*
_output_shapes

:22*
dtype0
dense_82/MatMulMatMuldense_81/Relu:activations:0&dense_82/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_82/BiasAdd/ReadVariableOpReadVariableOp(dense_82_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0
dense_82/BiasAddBiasAdddense_82/MatMul:product:0'dense_82/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2b
dense_82/ReluReludense_82/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_83/MatMul/ReadVariableOpReadVariableOp'dense_83_matmul_readvariableop_resource*
_output_shapes

:2*
dtype0
dense_83/MatMulMatMuldense_82/Relu:activations:0&dense_83/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
dense_83/BiasAdd/ReadVariableOpReadVariableOp(dense_83_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_83/BiasAddBiasAdddense_83/MatMul:product:0'dense_83/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџh
IdentityIdentitydense_83/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџЉ
NoOpNoOp ^dense_81/BiasAdd/ReadVariableOp^dense_81/MatMul/ReadVariableOp ^dense_82/BiasAdd/ReadVariableOp^dense_82/MatMul/ReadVariableOp ^dense_83/BiasAdd/ReadVariableOp^dense_83/MatMul/ReadVariableOp,^lstm_27/lstm_cell_27/BiasAdd/ReadVariableOp+^lstm_27/lstm_cell_27/MatMul/ReadVariableOp-^lstm_27/lstm_cell_27/MatMul_1/ReadVariableOp^lstm_27/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):џџџџџџџџџ: : : : : : : : : 2B
dense_81/BiasAdd/ReadVariableOpdense_81/BiasAdd/ReadVariableOp2@
dense_81/MatMul/ReadVariableOpdense_81/MatMul/ReadVariableOp2B
dense_82/BiasAdd/ReadVariableOpdense_82/BiasAdd/ReadVariableOp2@
dense_82/MatMul/ReadVariableOpdense_82/MatMul/ReadVariableOp2B
dense_83/BiasAdd/ReadVariableOpdense_83/BiasAdd/ReadVariableOp2@
dense_83/MatMul/ReadVariableOpdense_83/MatMul/ReadVariableOp2Z
+lstm_27/lstm_cell_27/BiasAdd/ReadVariableOp+lstm_27/lstm_cell_27/BiasAdd/ReadVariableOp2X
*lstm_27/lstm_cell_27/MatMul/ReadVariableOp*lstm_27/lstm_cell_27/MatMul/ReadVariableOp2\
,lstm_27/lstm_cell_27/MatMul_1/ReadVariableOp,lstm_27/lstm_cell_27/MatMul_1/ReadVariableOp2
lstm_27/whilelstm_27/while:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ј9
а
while_body_941540
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_27_matmul_readvariableop_resource_0:	H
5while_lstm_cell_27_matmul_1_readvariableop_resource_0:	dC
4while_lstm_cell_27_biasadd_readvariableop_resource_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_27_matmul_readvariableop_resource:	F
3while_lstm_cell_27_matmul_1_readvariableop_resource:	dA
2while_lstm_cell_27_biasadd_readvariableop_resource:	Ђ)while/lstm_cell_27/BiasAdd/ReadVariableOpЂ(while/lstm_cell_27/MatMul/ReadVariableOpЂ*while/lstm_cell_27/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   І
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype0
(while/lstm_cell_27/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_27_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype0К
while/lstm_cell_27/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_27/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЁ
*while/lstm_cell_27/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_27_matmul_1_readvariableop_resource_0*
_output_shapes
:	d*
dtype0Ё
while/lstm_cell_27/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_27/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ
while/lstm_cell_27/addAddV2#while/lstm_cell_27/MatMul:product:0%while/lstm_cell_27/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџ
)while/lstm_cell_27/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_27_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0Ї
while/lstm_cell_27/BiasAddBiasAddwhile/lstm_cell_27/add:z:01while/lstm_cell_27/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџd
"while/lstm_cell_27/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :я
while/lstm_cell_27/splitSplit+while/lstm_cell_27/split/split_dim:output:0#while/lstm_cell_27/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџd:џџџџџџџџџd:џџџџџџџџџd:џџџџџџџџџd*
	num_splitz
while/lstm_cell_27/SigmoidSigmoid!while/lstm_cell_27/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџd|
while/lstm_cell_27/Sigmoid_1Sigmoid!while/lstm_cell_27/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџd
while/lstm_cell_27/mulMul while/lstm_cell_27/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџdt
while/lstm_cell_27/ReluRelu!while/lstm_cell_27/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџd
while/lstm_cell_27/mul_1Mulwhile/lstm_cell_27/Sigmoid:y:0%while/lstm_cell_27/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџd
while/lstm_cell_27/add_1AddV2while/lstm_cell_27/mul:z:0while/lstm_cell_27/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџd|
while/lstm_cell_27/Sigmoid_2Sigmoid!while/lstm_cell_27/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџdq
while/lstm_cell_27/Relu_1Reluwhile/lstm_cell_27/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџd
while/lstm_cell_27/mul_2Mul while/lstm_cell_27/Sigmoid_2:y:0'while/lstm_cell_27/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџdr
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : э
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell_27/mul_2:z:0*
_output_shapes
: *
element_dtype0:щшвM
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
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: y
while/Identity_4Identitywhile/lstm_cell_27/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџdy
while/Identity_5Identitywhile/lstm_cell_27/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџdа

while/NoOpNoOp*^while/lstm_cell_27/BiasAdd/ReadVariableOp)^while/lstm_cell_27/MatMul/ReadVariableOp+^while/lstm_cell_27/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_27_biasadd_readvariableop_resource4while_lstm_cell_27_biasadd_readvariableop_resource_0"l
3while_lstm_cell_27_matmul_1_readvariableop_resource5while_lstm_cell_27_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_27_matmul_readvariableop_resource3while_lstm_cell_27_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџd:џџџџџџџџџd: : : : : 2V
)while/lstm_cell_27/BiasAdd/ReadVariableOp)while/lstm_cell_27/BiasAdd/ReadVariableOp2T
(while/lstm_cell_27/MatMul/ReadVariableOp(while/lstm_cell_27/MatMul/ReadVariableOp2X
*while/lstm_cell_27/MatMul_1/ReadVariableOp*while/lstm_cell_27/MatMul_1/ReadVariableOp: 
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
:џџџџџџџџџd:-)
'
_output_shapes
:џџџџџџџџџd:

_output_shapes
: :

_output_shapes
: 
т	
к
$__inference_signature_wrapper_940770
lstm_27_input
unknown:	
	unknown_0:	d
	unknown_1:	
	unknown_2:d2
	unknown_3:2
	unknown_4:22
	unknown_5:2
	unknown_6:2
	unknown_7:
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCalllstm_27_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*+
_read_only_resource_inputs
		*-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference__wrapped_model_939810o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):џџџџџџџџџ: : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
+
_output_shapes
:џџџџџџџџџ
'
_user_specified_namelstm_27_input
Т

)__inference_dense_82_layer_call_fn_941799

inputs
unknown:22
	unknown_0:2
identityЂStatefulPartitionedCallй
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_82_layer_call_and_return_conditional_losses_940351o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ2: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ2
 
_user_specified_nameinputs

З
(__inference_lstm_27_layer_call_fn_941168
inputs_0
unknown:	
	unknown_0:	d
	unknown_1:	
identityЂStatefulPartitionedCallч
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџd*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_lstm_27_layer_call_and_return_conditional_losses_940155o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџd`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџџџџџџџџџџ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/0
ИS

'sequential_27_lstm_27_while_body_939705H
Dsequential_27_lstm_27_while_sequential_27_lstm_27_while_loop_counterN
Jsequential_27_lstm_27_while_sequential_27_lstm_27_while_maximum_iterations+
'sequential_27_lstm_27_while_placeholder-
)sequential_27_lstm_27_while_placeholder_1-
)sequential_27_lstm_27_while_placeholder_2-
)sequential_27_lstm_27_while_placeholder_3G
Csequential_27_lstm_27_while_sequential_27_lstm_27_strided_slice_1_0
sequential_27_lstm_27_while_tensorarrayv2read_tensorlistgetitem_sequential_27_lstm_27_tensorarrayunstack_tensorlistfromtensor_0\
Isequential_27_lstm_27_while_lstm_cell_27_matmul_readvariableop_resource_0:	^
Ksequential_27_lstm_27_while_lstm_cell_27_matmul_1_readvariableop_resource_0:	dY
Jsequential_27_lstm_27_while_lstm_cell_27_biasadd_readvariableop_resource_0:	(
$sequential_27_lstm_27_while_identity*
&sequential_27_lstm_27_while_identity_1*
&sequential_27_lstm_27_while_identity_2*
&sequential_27_lstm_27_while_identity_3*
&sequential_27_lstm_27_while_identity_4*
&sequential_27_lstm_27_while_identity_5E
Asequential_27_lstm_27_while_sequential_27_lstm_27_strided_slice_1
}sequential_27_lstm_27_while_tensorarrayv2read_tensorlistgetitem_sequential_27_lstm_27_tensorarrayunstack_tensorlistfromtensorZ
Gsequential_27_lstm_27_while_lstm_cell_27_matmul_readvariableop_resource:	\
Isequential_27_lstm_27_while_lstm_cell_27_matmul_1_readvariableop_resource:	dW
Hsequential_27_lstm_27_while_lstm_cell_27_biasadd_readvariableop_resource:	Ђ?sequential_27/lstm_27/while/lstm_cell_27/BiasAdd/ReadVariableOpЂ>sequential_27/lstm_27/while/lstm_cell_27/MatMul/ReadVariableOpЂ@sequential_27/lstm_27/while/lstm_cell_27/MatMul_1/ReadVariableOp
Msequential_27/lstm_27/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   
?sequential_27/lstm_27/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsequential_27_lstm_27_while_tensorarrayv2read_tensorlistgetitem_sequential_27_lstm_27_tensorarrayunstack_tensorlistfromtensor_0'sequential_27_lstm_27_while_placeholderVsequential_27/lstm_27/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype0Щ
>sequential_27/lstm_27/while/lstm_cell_27/MatMul/ReadVariableOpReadVariableOpIsequential_27_lstm_27_while_lstm_cell_27_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype0ќ
/sequential_27/lstm_27/while/lstm_cell_27/MatMulMatMulFsequential_27/lstm_27/while/TensorArrayV2Read/TensorListGetItem:item:0Fsequential_27/lstm_27/while/lstm_cell_27/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЭ
@sequential_27/lstm_27/while/lstm_cell_27/MatMul_1/ReadVariableOpReadVariableOpKsequential_27_lstm_27_while_lstm_cell_27_matmul_1_readvariableop_resource_0*
_output_shapes
:	d*
dtype0у
1sequential_27/lstm_27/while/lstm_cell_27/MatMul_1MatMul)sequential_27_lstm_27_while_placeholder_2Hsequential_27/lstm_27/while/lstm_cell_27/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџр
,sequential_27/lstm_27/while/lstm_cell_27/addAddV29sequential_27/lstm_27/while/lstm_cell_27/MatMul:product:0;sequential_27/lstm_27/while/lstm_cell_27/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџЧ
?sequential_27/lstm_27/while/lstm_cell_27/BiasAdd/ReadVariableOpReadVariableOpJsequential_27_lstm_27_while_lstm_cell_27_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0щ
0sequential_27/lstm_27/while/lstm_cell_27/BiasAddBiasAdd0sequential_27/lstm_27/while/lstm_cell_27/add:z:0Gsequential_27/lstm_27/while/lstm_cell_27/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџz
8sequential_27/lstm_27/while/lstm_cell_27/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Б
.sequential_27/lstm_27/while/lstm_cell_27/splitSplitAsequential_27/lstm_27/while/lstm_cell_27/split/split_dim:output:09sequential_27/lstm_27/while/lstm_cell_27/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџd:џџџџџџџџџd:џџџџџџџџџd:џџџџџџџџџd*
	num_splitІ
0sequential_27/lstm_27/while/lstm_cell_27/SigmoidSigmoid7sequential_27/lstm_27/while/lstm_cell_27/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџdЈ
2sequential_27/lstm_27/while/lstm_cell_27/Sigmoid_1Sigmoid7sequential_27/lstm_27/while/lstm_cell_27/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџdШ
,sequential_27/lstm_27/while/lstm_cell_27/mulMul6sequential_27/lstm_27/while/lstm_cell_27/Sigmoid_1:y:0)sequential_27_lstm_27_while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџd 
-sequential_27/lstm_27/while/lstm_cell_27/ReluRelu7sequential_27/lstm_27/while/lstm_cell_27/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџdк
.sequential_27/lstm_27/while/lstm_cell_27/mul_1Mul4sequential_27/lstm_27/while/lstm_cell_27/Sigmoid:y:0;sequential_27/lstm_27/while/lstm_cell_27/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџdЯ
.sequential_27/lstm_27/while/lstm_cell_27/add_1AddV20sequential_27/lstm_27/while/lstm_cell_27/mul:z:02sequential_27/lstm_27/while/lstm_cell_27/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџdЈ
2sequential_27/lstm_27/while/lstm_cell_27/Sigmoid_2Sigmoid7sequential_27/lstm_27/while/lstm_cell_27/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџd
/sequential_27/lstm_27/while/lstm_cell_27/Relu_1Relu2sequential_27/lstm_27/while/lstm_cell_27/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџdо
.sequential_27/lstm_27/while/lstm_cell_27/mul_2Mul6sequential_27/lstm_27/while/lstm_cell_27/Sigmoid_2:y:0=sequential_27/lstm_27/while/lstm_cell_27/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџd
Fsequential_27/lstm_27/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : Х
@sequential_27/lstm_27/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem)sequential_27_lstm_27_while_placeholder_1Osequential_27/lstm_27/while/TensorArrayV2Write/TensorListSetItem/index:output:02sequential_27/lstm_27/while/lstm_cell_27/mul_2:z:0*
_output_shapes
: *
element_dtype0:щшвc
!sequential_27/lstm_27/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
sequential_27/lstm_27/while/addAddV2'sequential_27_lstm_27_while_placeholder*sequential_27/lstm_27/while/add/y:output:0*
T0*
_output_shapes
: e
#sequential_27/lstm_27/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :П
!sequential_27/lstm_27/while/add_1AddV2Dsequential_27_lstm_27_while_sequential_27_lstm_27_while_loop_counter,sequential_27/lstm_27/while/add_1/y:output:0*
T0*
_output_shapes
: 
$sequential_27/lstm_27/while/IdentityIdentity%sequential_27/lstm_27/while/add_1:z:0!^sequential_27/lstm_27/while/NoOp*
T0*
_output_shapes
: Т
&sequential_27/lstm_27/while/Identity_1IdentityJsequential_27_lstm_27_while_sequential_27_lstm_27_while_maximum_iterations!^sequential_27/lstm_27/while/NoOp*
T0*
_output_shapes
: 
&sequential_27/lstm_27/while/Identity_2Identity#sequential_27/lstm_27/while/add:z:0!^sequential_27/lstm_27/while/NoOp*
T0*
_output_shapes
: Ш
&sequential_27/lstm_27/while/Identity_3IdentityPsequential_27/lstm_27/while/TensorArrayV2Write/TensorListSetItem:output_handle:0!^sequential_27/lstm_27/while/NoOp*
T0*
_output_shapes
: Л
&sequential_27/lstm_27/while/Identity_4Identity2sequential_27/lstm_27/while/lstm_cell_27/mul_2:z:0!^sequential_27/lstm_27/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџdЛ
&sequential_27/lstm_27/while/Identity_5Identity2sequential_27/lstm_27/while/lstm_cell_27/add_1:z:0!^sequential_27/lstm_27/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџdЈ
 sequential_27/lstm_27/while/NoOpNoOp@^sequential_27/lstm_27/while/lstm_cell_27/BiasAdd/ReadVariableOp?^sequential_27/lstm_27/while/lstm_cell_27/MatMul/ReadVariableOpA^sequential_27/lstm_27/while/lstm_cell_27/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "U
$sequential_27_lstm_27_while_identity-sequential_27/lstm_27/while/Identity:output:0"Y
&sequential_27_lstm_27_while_identity_1/sequential_27/lstm_27/while/Identity_1:output:0"Y
&sequential_27_lstm_27_while_identity_2/sequential_27/lstm_27/while/Identity_2:output:0"Y
&sequential_27_lstm_27_while_identity_3/sequential_27/lstm_27/while/Identity_3:output:0"Y
&sequential_27_lstm_27_while_identity_4/sequential_27/lstm_27/while/Identity_4:output:0"Y
&sequential_27_lstm_27_while_identity_5/sequential_27/lstm_27/while/Identity_5:output:0"
Hsequential_27_lstm_27_while_lstm_cell_27_biasadd_readvariableop_resourceJsequential_27_lstm_27_while_lstm_cell_27_biasadd_readvariableop_resource_0"
Isequential_27_lstm_27_while_lstm_cell_27_matmul_1_readvariableop_resourceKsequential_27_lstm_27_while_lstm_cell_27_matmul_1_readvariableop_resource_0"
Gsequential_27_lstm_27_while_lstm_cell_27_matmul_readvariableop_resourceIsequential_27_lstm_27_while_lstm_cell_27_matmul_readvariableop_resource_0"
Asequential_27_lstm_27_while_sequential_27_lstm_27_strided_slice_1Csequential_27_lstm_27_while_sequential_27_lstm_27_strided_slice_1_0"
}sequential_27_lstm_27_while_tensorarrayv2read_tensorlistgetitem_sequential_27_lstm_27_tensorarrayunstack_tensorlistfromtensorsequential_27_lstm_27_while_tensorarrayv2read_tensorlistgetitem_sequential_27_lstm_27_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџd:џџџџџџџџџd: : : : : 2
?sequential_27/lstm_27/while/lstm_cell_27/BiasAdd/ReadVariableOp?sequential_27/lstm_27/while/lstm_cell_27/BiasAdd/ReadVariableOp2
>sequential_27/lstm_27/while/lstm_cell_27/MatMul/ReadVariableOp>sequential_27/lstm_27/while/lstm_cell_27/MatMul/ReadVariableOp2
@sequential_27/lstm_27/while/lstm_cell_27/MatMul_1/ReadVariableOp@sequential_27/lstm_27/while/lstm_cell_27/MatMul_1/ReadVariableOp: 
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
:џџџџџџџџџd:-)
'
_output_shapes
:џџџџџџџџџd:

_output_shapes
: :

_output_shapes
: 

в
"__inference__traced_restore_942176
file_prefix2
 assignvariableop_dense_81_kernel:d2.
 assignvariableop_1_dense_81_bias:24
"assignvariableop_2_dense_82_kernel:22.
 assignvariableop_3_dense_82_bias:24
"assignvariableop_4_dense_83_kernel:2.
 assignvariableop_5_dense_83_bias:A
.assignvariableop_6_lstm_27_lstm_cell_27_kernel:	K
8assignvariableop_7_lstm_27_lstm_cell_27_recurrent_kernel:	d;
,assignvariableop_8_lstm_27_lstm_cell_27_bias:	&
assignvariableop_9_adam_iter:	 )
assignvariableop_10_adam_beta_1: )
assignvariableop_11_adam_beta_2: (
assignvariableop_12_adam_decay: 0
&assignvariableop_13_adam_learning_rate: %
assignvariableop_14_total_1: %
assignvariableop_15_count_1: #
assignvariableop_16_total: #
assignvariableop_17_count: <
*assignvariableop_18_adam_dense_81_kernel_m:d26
(assignvariableop_19_adam_dense_81_bias_m:2<
*assignvariableop_20_adam_dense_82_kernel_m:226
(assignvariableop_21_adam_dense_82_bias_m:2<
*assignvariableop_22_adam_dense_83_kernel_m:26
(assignvariableop_23_adam_dense_83_bias_m:I
6assignvariableop_24_adam_lstm_27_lstm_cell_27_kernel_m:	S
@assignvariableop_25_adam_lstm_27_lstm_cell_27_recurrent_kernel_m:	dC
4assignvariableop_26_adam_lstm_27_lstm_cell_27_bias_m:	<
*assignvariableop_27_adam_dense_81_kernel_v:d26
(assignvariableop_28_adam_dense_81_bias_v:2<
*assignvariableop_29_adam_dense_82_kernel_v:226
(assignvariableop_30_adam_dense_82_bias_v:2<
*assignvariableop_31_adam_dense_83_kernel_v:26
(assignvariableop_32_adam_dense_83_bias_v:I
6assignvariableop_33_adam_lstm_27_lstm_cell_27_kernel_v:	S
@assignvariableop_34_adam_lstm_27_lstm_cell_27_recurrent_kernel_v:	dC
4assignvariableop_35_adam_lstm_27_lstm_cell_27_bias_v:	
identity_37ЂAssignVariableOpЂAssignVariableOp_1ЂAssignVariableOp_10ЂAssignVariableOp_11ЂAssignVariableOp_12ЂAssignVariableOp_13ЂAssignVariableOp_14ЂAssignVariableOp_15ЂAssignVariableOp_16ЂAssignVariableOp_17ЂAssignVariableOp_18ЂAssignVariableOp_19ЂAssignVariableOp_2ЂAssignVariableOp_20ЂAssignVariableOp_21ЂAssignVariableOp_22ЂAssignVariableOp_23ЂAssignVariableOp_24ЂAssignVariableOp_25ЂAssignVariableOp_26ЂAssignVariableOp_27ЂAssignVariableOp_28ЂAssignVariableOp_29ЂAssignVariableOp_3ЂAssignVariableOp_30ЂAssignVariableOp_31ЂAssignVariableOp_32ЂAssignVariableOp_33ЂAssignVariableOp_34ЂAssignVariableOp_35ЂAssignVariableOp_4ЂAssignVariableOp_5ЂAssignVariableOp_6ЂAssignVariableOp_7ЂAssignVariableOp_8ЂAssignVariableOp_9
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:%*
dtype0*Ў
valueЄBЁ%B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHК
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:%*
dtype0*]
valueTBR%B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B к
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Њ
_output_shapes
:::::::::::::::::::::::::::::::::::::*3
dtypes)
'2%	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOp assignvariableop_dense_81_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_81_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOp"assignvariableop_2_dense_82_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOp assignvariableop_3_dense_82_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOp"assignvariableop_4_dense_83_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOp assignvariableop_5_dense_83_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOp.assignvariableop_6_lstm_27_lstm_cell_27_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:Ї
AssignVariableOp_7AssignVariableOp8assignvariableop_7_lstm_27_lstm_cell_27_recurrent_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_8AssignVariableOp,assignvariableop_8_lstm_27_lstm_cell_27_biasIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_iterIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_beta_1Identity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_beta_2Identity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_12AssignVariableOpassignvariableop_12_adam_decayIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOp&assignvariableop_13_adam_learning_rateIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_14AssignVariableOpassignvariableop_14_total_1Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_15AssignVariableOpassignvariableop_15_count_1Identity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_16AssignVariableOpassignvariableop_16_totalIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_17AssignVariableOpassignvariableop_17_countIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_18AssignVariableOp*assignvariableop_18_adam_dense_81_kernel_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_19AssignVariableOp(assignvariableop_19_adam_dense_81_bias_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_20AssignVariableOp*assignvariableop_20_adam_dense_82_kernel_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_21AssignVariableOp(assignvariableop_21_adam_dense_82_bias_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_22AssignVariableOp*assignvariableop_22_adam_dense_83_kernel_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_23AssignVariableOp(assignvariableop_23_adam_dense_83_bias_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:Ї
AssignVariableOp_24AssignVariableOp6assignvariableop_24_adam_lstm_27_lstm_cell_27_kernel_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:Б
AssignVariableOp_25AssignVariableOp@assignvariableop_25_adam_lstm_27_lstm_cell_27_recurrent_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:Ѕ
AssignVariableOp_26AssignVariableOp4assignvariableop_26_adam_lstm_27_lstm_cell_27_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_27AssignVariableOp*assignvariableop_27_adam_dense_81_kernel_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_28AssignVariableOp(assignvariableop_28_adam_dense_81_bias_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_29AssignVariableOp*assignvariableop_29_adam_dense_82_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_30AssignVariableOp(assignvariableop_30_adam_dense_82_bias_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_31AssignVariableOp*assignvariableop_31_adam_dense_83_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_32AssignVariableOp(assignvariableop_32_adam_dense_83_bias_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:Ї
AssignVariableOp_33AssignVariableOp6assignvariableop_33_adam_lstm_27_lstm_cell_27_kernel_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:Б
AssignVariableOp_34AssignVariableOp@assignvariableop_34_adam_lstm_27_lstm_cell_27_recurrent_kernel_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:Ѕ
AssignVariableOp_35AssignVariableOp4assignvariableop_35_adam_lstm_27_lstm_cell_27_bias_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ч
Identity_36Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_37IdentityIdentity_36:output:0^NoOp_1*
T0*
_output_shapes
: д
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


ѕ
D__inference_dense_82_layer_call_and_return_conditional_losses_940351

inputs0
matmul_readvariableop_resource:22-
biasadd_readvariableop_resource:2
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:22*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ2: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ2
 
_user_specified_nameinputs
ї
Е
(__inference_lstm_27_layer_call_fn_941179

inputs
unknown:	
	unknown_0:	d
	unknown_1:	
identityЂStatefulPartitionedCallх
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџd*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_lstm_27_layer_call_and_return_conditional_losses_940315o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџd`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:џџџџџџџџџ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
$
у
while_body_940085
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0.
while_lstm_cell_27_940109_0:	.
while_lstm_cell_27_940111_0:	d*
while_lstm_cell_27_940113_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor,
while_lstm_cell_27_940109:	,
while_lstm_cell_27_940111:	d(
while_lstm_cell_27_940113:	Ђ*while/lstm_cell_27/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   І
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype0Г
*while/lstm_cell_27/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_27_940109_0while_lstm_cell_27_940111_0while_lstm_cell_27_940113_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:џџџџџџџџџd:џџџџџџџџџd:џџџџџџџџџd*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_lstm_cell_27_layer_call_and_return_conditional_losses_940025r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : 
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:03while/lstm_cell_27/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:щшвM
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
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_4Identity3while/lstm_cell_27/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџd
while/Identity_5Identity3while/lstm_cell_27/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџdy

while/NoOpNoOp+^while/lstm_cell_27/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"8
while_lstm_cell_27_940109while_lstm_cell_27_940109_0"8
while_lstm_cell_27_940111while_lstm_cell_27_940111_0"8
while_lstm_cell_27_940113while_lstm_cell_27_940113_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџd:џџџџџџџџџd: : : : : 2X
*while/lstm_cell_27/StatefulPartitionedCall*while/lstm_cell_27/StatefulPartitionedCall: 
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
:џџџџџџџџџd:-)
'
_output_shapes
:џџџџџџџџџd:

_output_shapes
: :

_output_shapes
: 
н

H__inference_lstm_cell_27_layer_call_and_return_conditional_losses_941927

inputs
states_0
states_11
matmul_readvariableop_resource:	3
 matmul_1_readvariableop_resource:	d.
biasadd_readvariableop_resource:	
identity

identity_1

identity_2ЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpЂMatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџy
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	d*
dtype0p
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ж
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџd:џџџџџџџџџd:џџџџџџџџџd:џџџџџџџџџd*
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:џџџџџџџџџdV
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:џџџџџџџџџdU
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:џџџџџџџџџdN
ReluRelusplit:output:2*
T0*'
_output_shapes
:џџџџџџџџџd_
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџdT
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџdV
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:џџџџџџџџџdK
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџdc
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџdX
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџdZ

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџdZ

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџd
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:џџџџџџџџџ:џџџџџџџџџd:џџџџџџџџџd: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:QM
'
_output_shapes
:џџџџџџџџџd
"
_user_specified_name
states/0:QM
'
_output_shapes
:џџџџџџџџџd
"
_user_specified_name
states/1
Е
У
while_cond_940229
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_940229___redundant_placeholder04
0while_while_cond_940229___redundant_placeholder14
0while_while_cond_940229___redundant_placeholder24
0while_while_cond_940229___redundant_placeholder3
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
@: : : : :џџџџџџџџџd:џџџџџџџџџd: ::::: 
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
:џџџџџџџџџd:-)
'
_output_shapes
:џџџџџџџџџd:

_output_shapes
: :

_output_shapes
:


ѕ
D__inference_dense_81_layer_call_and_return_conditional_losses_941790

inputs0
matmul_readvariableop_resource:d2-
biasadd_readvariableop_resource:2
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d2*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџd: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџd
 
_user_specified_nameinputs
И
Ѕ
I__inference_sequential_27_layer_call_and_return_conditional_losses_940739
lstm_27_input!
lstm_27_940716:	!
lstm_27_940718:	d
lstm_27_940720:	!
dense_81_940723:d2
dense_81_940725:2!
dense_82_940728:22
dense_82_940730:2!
dense_83_940733:2
dense_83_940735:
identityЂ dense_81/StatefulPartitionedCallЂ dense_82/StatefulPartitionedCallЂ dense_83/StatefulPartitionedCallЂlstm_27/StatefulPartitionedCall
lstm_27/StatefulPartitionedCallStatefulPartitionedCalllstm_27_inputlstm_27_940716lstm_27_940718lstm_27_940720*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџd*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_lstm_27_layer_call_and_return_conditional_losses_940583
 dense_81/StatefulPartitionedCallStatefulPartitionedCall(lstm_27/StatefulPartitionedCall:output:0dense_81_940723dense_81_940725*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_81_layer_call_and_return_conditional_losses_940334
 dense_82/StatefulPartitionedCallStatefulPartitionedCall)dense_81/StatefulPartitionedCall:output:0dense_82_940728dense_82_940730*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_82_layer_call_and_return_conditional_losses_940351
 dense_83/StatefulPartitionedCallStatefulPartitionedCall)dense_82/StatefulPartitionedCall:output:0dense_83_940733dense_83_940735*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_83_layer_call_and_return_conditional_losses_940367x
IdentityIdentity)dense_83/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџб
NoOpNoOp!^dense_81/StatefulPartitionedCall!^dense_82/StatefulPartitionedCall!^dense_83/StatefulPartitionedCall ^lstm_27/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):џџџџџџџџџ: : : : : : : : : 2D
 dense_81/StatefulPartitionedCall dense_81/StatefulPartitionedCall2D
 dense_82/StatefulPartitionedCall dense_82/StatefulPartitionedCall2D
 dense_83/StatefulPartitionedCall dense_83/StatefulPartitionedCall2B
lstm_27/StatefulPartitionedCalllstm_27/StatefulPartitionedCall:Z V
+
_output_shapes
:џџџџџџџџџ
'
_user_specified_namelstm_27_input
Т

)__inference_dense_83_layer_call_fn_941819

inputs
unknown:2
	unknown_0:
identityЂStatefulPartitionedCallй
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_83_layer_call_and_return_conditional_losses_940367o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ2: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ2
 
_user_specified_nameinputs
Е
У
while_cond_941539
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_941539___redundant_placeholder04
0while_while_cond_941539___redundant_placeholder14
0while_while_cond_941539___redundant_placeholder24
0while_while_cond_941539___redundant_placeholder3
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
@: : : : :џџџџџџџџџd:џџџџџџџџџd: ::::: 
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
:џџџџџџџџџd:-)
'
_output_shapes
:џџџџџџџџџd:

_output_shapes
: :

_output_shapes
:
е

H__inference_lstm_cell_27_layer_call_and_return_conditional_losses_939877

inputs

states
states_11
matmul_readvariableop_resource:	3
 matmul_1_readvariableop_resource:	d.
biasadd_readvariableop_resource:	
identity

identity_1

identity_2ЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpЂMatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџy
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	d*
dtype0n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ж
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџd:џџџџџџџџџd:џџџџџџџџџd:џџџџџџџџџd*
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:џџџџџџџџџdV
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:џџџџџџџџџdU
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:џџџџџџџџџdN
ReluRelusplit:output:2*
T0*'
_output_shapes
:џџџџџџџџџd_
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџdT
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџdV
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:џџџџџџџџџdK
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџdc
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџdX
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџdZ

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџdZ

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџd
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:џџџџџџџџџ:џџџџџџџџџd:џџџџџџџџџd: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџd
 
_user_specified_namestates:OK
'
_output_shapes
:џџџџџџџџџd
 
_user_specified_namestates
н

H__inference_lstm_cell_27_layer_call_and_return_conditional_losses_941895

inputs
states_0
states_11
matmul_readvariableop_resource:	3
 matmul_1_readvariableop_resource:	d.
biasadd_readvariableop_resource:	
identity

identity_1

identity_2ЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpЂMatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџy
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	d*
dtype0p
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ж
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџd:џџџџџџџџџd:џџџџџџџџџd:џџџџџџџџџd*
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:џџџџџџџџџdV
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:џџџџџџџџџdU
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:џџџџџџџџџdN
ReluRelusplit:output:2*
T0*'
_output_shapes
:џџџџџџџџџd_
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџdT
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџdV
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:џџџџџџџџџdK
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџdc
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџdX
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџdZ

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџdZ

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџd
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:џџџџџџџџџ:џџџџџџџџџd:џџџџџџџџџd: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:QM
'
_output_shapes
:џџџџџџџџџd
"
_user_specified_name
states/0:QM
'
_output_shapes
:џџџџџџџџџd
"
_user_specified_name
states/1
Ѓ

I__inference_sequential_27_layer_call_and_return_conditional_losses_940643

inputs!
lstm_27_940620:	!
lstm_27_940622:	d
lstm_27_940624:	!
dense_81_940627:d2
dense_81_940629:2!
dense_82_940632:22
dense_82_940634:2!
dense_83_940637:2
dense_83_940639:
identityЂ dense_81/StatefulPartitionedCallЂ dense_82/StatefulPartitionedCallЂ dense_83/StatefulPartitionedCallЂlstm_27/StatefulPartitionedCallў
lstm_27/StatefulPartitionedCallStatefulPartitionedCallinputslstm_27_940620lstm_27_940622lstm_27_940624*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџd*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_lstm_27_layer_call_and_return_conditional_losses_940583
 dense_81/StatefulPartitionedCallStatefulPartitionedCall(lstm_27/StatefulPartitionedCall:output:0dense_81_940627dense_81_940629*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_81_layer_call_and_return_conditional_losses_940334
 dense_82/StatefulPartitionedCallStatefulPartitionedCall)dense_81/StatefulPartitionedCall:output:0dense_82_940632dense_82_940634*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_82_layer_call_and_return_conditional_losses_940351
 dense_83/StatefulPartitionedCallStatefulPartitionedCall)dense_82/StatefulPartitionedCall:output:0dense_83_940637dense_83_940639*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_83_layer_call_and_return_conditional_losses_940367x
IdentityIdentity)dense_83/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџб
NoOpNoOp!^dense_81/StatefulPartitionedCall!^dense_82/StatefulPartitionedCall!^dense_83/StatefulPartitionedCall ^lstm_27/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):џџџџџџџџџ: : : : : : : : : 2D
 dense_81/StatefulPartitionedCall dense_81/StatefulPartitionedCall2D
 dense_82/StatefulPartitionedCall dense_82/StatefulPartitionedCall2D
 dense_83/StatefulPartitionedCall dense_83/StatefulPartitionedCall2B
lstm_27/StatefulPartitionedCalllstm_27/StatefulPartitionedCall:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ђ
ф	
!__inference__wrapped_model_939810
lstm_27_inputT
Asequential_27_lstm_27_lstm_cell_27_matmul_readvariableop_resource:	V
Csequential_27_lstm_27_lstm_cell_27_matmul_1_readvariableop_resource:	dQ
Bsequential_27_lstm_27_lstm_cell_27_biasadd_readvariableop_resource:	G
5sequential_27_dense_81_matmul_readvariableop_resource:d2D
6sequential_27_dense_81_biasadd_readvariableop_resource:2G
5sequential_27_dense_82_matmul_readvariableop_resource:22D
6sequential_27_dense_82_biasadd_readvariableop_resource:2G
5sequential_27_dense_83_matmul_readvariableop_resource:2D
6sequential_27_dense_83_biasadd_readvariableop_resource:
identityЂ-sequential_27/dense_81/BiasAdd/ReadVariableOpЂ,sequential_27/dense_81/MatMul/ReadVariableOpЂ-sequential_27/dense_82/BiasAdd/ReadVariableOpЂ,sequential_27/dense_82/MatMul/ReadVariableOpЂ-sequential_27/dense_83/BiasAdd/ReadVariableOpЂ,sequential_27/dense_83/MatMul/ReadVariableOpЂ9sequential_27/lstm_27/lstm_cell_27/BiasAdd/ReadVariableOpЂ8sequential_27/lstm_27/lstm_cell_27/MatMul/ReadVariableOpЂ:sequential_27/lstm_27/lstm_cell_27/MatMul_1/ReadVariableOpЂsequential_27/lstm_27/whileX
sequential_27/lstm_27/ShapeShapelstm_27_input*
T0*
_output_shapes
:s
)sequential_27/lstm_27/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+sequential_27/lstm_27/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+sequential_27/lstm_27/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:П
#sequential_27/lstm_27/strided_sliceStridedSlice$sequential_27/lstm_27/Shape:output:02sequential_27/lstm_27/strided_slice/stack:output:04sequential_27/lstm_27/strided_slice/stack_1:output:04sequential_27/lstm_27/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
$sequential_27/lstm_27/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :dЕ
"sequential_27/lstm_27/zeros/packedPack,sequential_27/lstm_27/strided_slice:output:0-sequential_27/lstm_27/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:f
!sequential_27/lstm_27/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Ў
sequential_27/lstm_27/zerosFill+sequential_27/lstm_27/zeros/packed:output:0*sequential_27/lstm_27/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџdh
&sequential_27/lstm_27/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :dЙ
$sequential_27/lstm_27/zeros_1/packedPack,sequential_27/lstm_27/strided_slice:output:0/sequential_27/lstm_27/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:h
#sequential_27/lstm_27/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Д
sequential_27/lstm_27/zeros_1Fill-sequential_27/lstm_27/zeros_1/packed:output:0,sequential_27/lstm_27/zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџdy
$sequential_27/lstm_27/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"           
sequential_27/lstm_27/transpose	Transposelstm_27_input-sequential_27/lstm_27/transpose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџp
sequential_27/lstm_27/Shape_1Shape#sequential_27/lstm_27/transpose:y:0*
T0*
_output_shapes
:u
+sequential_27/lstm_27/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-sequential_27/lstm_27/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-sequential_27/lstm_27/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Щ
%sequential_27/lstm_27/strided_slice_1StridedSlice&sequential_27/lstm_27/Shape_1:output:04sequential_27/lstm_27/strided_slice_1/stack:output:06sequential_27/lstm_27/strided_slice_1/stack_1:output:06sequential_27/lstm_27/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask|
1sequential_27/lstm_27/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџі
#sequential_27/lstm_27/TensorArrayV2TensorListReserve:sequential_27/lstm_27/TensorArrayV2/element_shape:output:0.sequential_27/lstm_27/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
Ksequential_27/lstm_27/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   Ђ
=sequential_27/lstm_27/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor#sequential_27/lstm_27/transpose:y:0Tsequential_27/lstm_27/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвu
+sequential_27/lstm_27/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-sequential_27/lstm_27/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-sequential_27/lstm_27/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:з
%sequential_27/lstm_27/strided_slice_2StridedSlice#sequential_27/lstm_27/transpose:y:04sequential_27/lstm_27/strided_slice_2/stack:output:06sequential_27/lstm_27/strided_slice_2/stack_1:output:06sequential_27/lstm_27/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_maskЛ
8sequential_27/lstm_27/lstm_cell_27/MatMul/ReadVariableOpReadVariableOpAsequential_27_lstm_27_lstm_cell_27_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0и
)sequential_27/lstm_27/lstm_cell_27/MatMulMatMul.sequential_27/lstm_27/strided_slice_2:output:0@sequential_27/lstm_27/lstm_cell_27/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџП
:sequential_27/lstm_27/lstm_cell_27/MatMul_1/ReadVariableOpReadVariableOpCsequential_27_lstm_27_lstm_cell_27_matmul_1_readvariableop_resource*
_output_shapes
:	d*
dtype0в
+sequential_27/lstm_27/lstm_cell_27/MatMul_1MatMul$sequential_27/lstm_27/zeros:output:0Bsequential_27/lstm_27/lstm_cell_27/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЮ
&sequential_27/lstm_27/lstm_cell_27/addAddV23sequential_27/lstm_27/lstm_cell_27/MatMul:product:05sequential_27/lstm_27/lstm_cell_27/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџЙ
9sequential_27/lstm_27/lstm_cell_27/BiasAdd/ReadVariableOpReadVariableOpBsequential_27_lstm_27_lstm_cell_27_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0з
*sequential_27/lstm_27/lstm_cell_27/BiasAddBiasAdd*sequential_27/lstm_27/lstm_cell_27/add:z:0Asequential_27/lstm_27/lstm_cell_27/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџt
2sequential_27/lstm_27/lstm_cell_27/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
(sequential_27/lstm_27/lstm_cell_27/splitSplit;sequential_27/lstm_27/lstm_cell_27/split/split_dim:output:03sequential_27/lstm_27/lstm_cell_27/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџd:џџџџџџџџџd:џџџџџџџџџd:џџџџџџџџџd*
	num_split
*sequential_27/lstm_27/lstm_cell_27/SigmoidSigmoid1sequential_27/lstm_27/lstm_cell_27/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџd
,sequential_27/lstm_27/lstm_cell_27/Sigmoid_1Sigmoid1sequential_27/lstm_27/lstm_cell_27/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџdЙ
&sequential_27/lstm_27/lstm_cell_27/mulMul0sequential_27/lstm_27/lstm_cell_27/Sigmoid_1:y:0&sequential_27/lstm_27/zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџd
'sequential_27/lstm_27/lstm_cell_27/ReluRelu1sequential_27/lstm_27/lstm_cell_27/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџdШ
(sequential_27/lstm_27/lstm_cell_27/mul_1Mul.sequential_27/lstm_27/lstm_cell_27/Sigmoid:y:05sequential_27/lstm_27/lstm_cell_27/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџdН
(sequential_27/lstm_27/lstm_cell_27/add_1AddV2*sequential_27/lstm_27/lstm_cell_27/mul:z:0,sequential_27/lstm_27/lstm_cell_27/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџd
,sequential_27/lstm_27/lstm_cell_27/Sigmoid_2Sigmoid1sequential_27/lstm_27/lstm_cell_27/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџd
)sequential_27/lstm_27/lstm_cell_27/Relu_1Relu,sequential_27/lstm_27/lstm_cell_27/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџdЬ
(sequential_27/lstm_27/lstm_cell_27/mul_2Mul0sequential_27/lstm_27/lstm_cell_27/Sigmoid_2:y:07sequential_27/lstm_27/lstm_cell_27/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџd
3sequential_27/lstm_27/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџd   t
2sequential_27/lstm_27/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :
%sequential_27/lstm_27/TensorArrayV2_1TensorListReserve<sequential_27/lstm_27/TensorArrayV2_1/element_shape:output:0;sequential_27/lstm_27/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв\
sequential_27/lstm_27/timeConst*
_output_shapes
: *
dtype0*
value	B : y
.sequential_27/lstm_27/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџj
(sequential_27/lstm_27/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ж
sequential_27/lstm_27/whileWhile1sequential_27/lstm_27/while/loop_counter:output:07sequential_27/lstm_27/while/maximum_iterations:output:0#sequential_27/lstm_27/time:output:0.sequential_27/lstm_27/TensorArrayV2_1:handle:0$sequential_27/lstm_27/zeros:output:0&sequential_27/lstm_27/zeros_1:output:0.sequential_27/lstm_27/strided_slice_1:output:0Msequential_27/lstm_27/TensorArrayUnstack/TensorListFromTensor:output_handle:0Asequential_27_lstm_27_lstm_cell_27_matmul_readvariableop_resourceCsequential_27_lstm_27_lstm_cell_27_matmul_1_readvariableop_resourceBsequential_27_lstm_27_lstm_cell_27_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџd:џџџџџџџџџd: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *3
body+R)
'sequential_27_lstm_27_while_body_939705*3
cond+R)
'sequential_27_lstm_27_while_cond_939704*K
output_shapes:
8: : : : :џџџџџџџџџd:џџџџџџџџџd: : : : : *
parallel_iterations 
Fsequential_27/lstm_27/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџd   
8sequential_27/lstm_27/TensorArrayV2Stack/TensorListStackTensorListStack$sequential_27/lstm_27/while:output:3Osequential_27/lstm_27/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџd*
element_dtype0*
num_elements~
+sequential_27/lstm_27/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџw
-sequential_27/lstm_27/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: w
-sequential_27/lstm_27/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ѕ
%sequential_27/lstm_27/strided_slice_3StridedSliceAsequential_27/lstm_27/TensorArrayV2Stack/TensorListStack:tensor:04sequential_27/lstm_27/strided_slice_3/stack:output:06sequential_27/lstm_27/strided_slice_3/stack_1:output:06sequential_27/lstm_27/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџd*
shrink_axis_mask{
&sequential_27/lstm_27/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          и
!sequential_27/lstm_27/transpose_1	TransposeAsequential_27/lstm_27/TensorArrayV2Stack/TensorListStack:tensor:0/sequential_27/lstm_27/transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџdq
sequential_27/lstm_27/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    Ђ
,sequential_27/dense_81/MatMul/ReadVariableOpReadVariableOp5sequential_27_dense_81_matmul_readvariableop_resource*
_output_shapes

:d2*
dtype0П
sequential_27/dense_81/MatMulMatMul.sequential_27/lstm_27/strided_slice_3:output:04sequential_27/dense_81/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2 
-sequential_27/dense_81/BiasAdd/ReadVariableOpReadVariableOp6sequential_27_dense_81_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0Л
sequential_27/dense_81/BiasAddBiasAdd'sequential_27/dense_81/MatMul:product:05sequential_27/dense_81/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2~
sequential_27/dense_81/ReluRelu'sequential_27/dense_81/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2Ђ
,sequential_27/dense_82/MatMul/ReadVariableOpReadVariableOp5sequential_27_dense_82_matmul_readvariableop_resource*
_output_shapes

:22*
dtype0К
sequential_27/dense_82/MatMulMatMul)sequential_27/dense_81/Relu:activations:04sequential_27/dense_82/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2 
-sequential_27/dense_82/BiasAdd/ReadVariableOpReadVariableOp6sequential_27_dense_82_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0Л
sequential_27/dense_82/BiasAddBiasAdd'sequential_27/dense_82/MatMul:product:05sequential_27/dense_82/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2~
sequential_27/dense_82/ReluRelu'sequential_27/dense_82/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2Ђ
,sequential_27/dense_83/MatMul/ReadVariableOpReadVariableOp5sequential_27_dense_83_matmul_readvariableop_resource*
_output_shapes

:2*
dtype0К
sequential_27/dense_83/MatMulMatMul)sequential_27/dense_82/Relu:activations:04sequential_27/dense_83/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 
-sequential_27/dense_83/BiasAdd/ReadVariableOpReadVariableOp6sequential_27_dense_83_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Л
sequential_27/dense_83/BiasAddBiasAdd'sequential_27/dense_83/MatMul:product:05sequential_27/dense_83/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџv
IdentityIdentity'sequential_27/dense_83/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџЕ
NoOpNoOp.^sequential_27/dense_81/BiasAdd/ReadVariableOp-^sequential_27/dense_81/MatMul/ReadVariableOp.^sequential_27/dense_82/BiasAdd/ReadVariableOp-^sequential_27/dense_82/MatMul/ReadVariableOp.^sequential_27/dense_83/BiasAdd/ReadVariableOp-^sequential_27/dense_83/MatMul/ReadVariableOp:^sequential_27/lstm_27/lstm_cell_27/BiasAdd/ReadVariableOp9^sequential_27/lstm_27/lstm_cell_27/MatMul/ReadVariableOp;^sequential_27/lstm_27/lstm_cell_27/MatMul_1/ReadVariableOp^sequential_27/lstm_27/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):џџџџџџџџџ: : : : : : : : : 2^
-sequential_27/dense_81/BiasAdd/ReadVariableOp-sequential_27/dense_81/BiasAdd/ReadVariableOp2\
,sequential_27/dense_81/MatMul/ReadVariableOp,sequential_27/dense_81/MatMul/ReadVariableOp2^
-sequential_27/dense_82/BiasAdd/ReadVariableOp-sequential_27/dense_82/BiasAdd/ReadVariableOp2\
,sequential_27/dense_82/MatMul/ReadVariableOp,sequential_27/dense_82/MatMul/ReadVariableOp2^
-sequential_27/dense_83/BiasAdd/ReadVariableOp-sequential_27/dense_83/BiasAdd/ReadVariableOp2\
,sequential_27/dense_83/MatMul/ReadVariableOp,sequential_27/dense_83/MatMul/ReadVariableOp2v
9sequential_27/lstm_27/lstm_cell_27/BiasAdd/ReadVariableOp9sequential_27/lstm_27/lstm_cell_27/BiasAdd/ReadVariableOp2t
8sequential_27/lstm_27/lstm_cell_27/MatMul/ReadVariableOp8sequential_27/lstm_27/lstm_cell_27/MatMul/ReadVariableOp2x
:sequential_27/lstm_27/lstm_cell_27/MatMul_1/ReadVariableOp:sequential_27/lstm_27/lstm_cell_27/MatMul_1/ReadVariableOp2:
sequential_27/lstm_27/whilesequential_27/lstm_27/while:Z V
+
_output_shapes
:џџџџџџџџџ
'
_user_specified_namelstm_27_input


ѕ
D__inference_dense_82_layer_call_and_return_conditional_losses_941810

inputs0
matmul_readvariableop_resource:22-
biasadd_readvariableop_resource:2
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:22*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ2: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ2
 
_user_specified_nameinputs
Е
У
while_cond_941684
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_941684___redundant_placeholder04
0while_while_cond_941684___redundant_placeholder14
0while_while_cond_941684___redundant_placeholder24
0while_while_cond_941684___redundant_placeholder3
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
@: : : : :џџџџџџџџџd:џџџџџџџџџd: ::::: 
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
:џџџџџџџџџd:-)
'
_output_shapes
:џџџџџџџџџd:

_output_shapes
: :

_output_shapes
:
ї
Е
(__inference_lstm_27_layer_call_fn_941190

inputs
unknown:	
	unknown_0:	d
	unknown_1:	
identityЂStatefulPartitionedCallх
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџd*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_lstm_27_layer_call_and_return_conditional_losses_940583o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџd`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:џџџџџџџџџ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ЧL
ш
__inference__traced_save_942058
file_prefix.
*savev2_dense_81_kernel_read_readvariableop,
(savev2_dense_81_bias_read_readvariableop.
*savev2_dense_82_kernel_read_readvariableop,
(savev2_dense_82_bias_read_readvariableop.
*savev2_dense_83_kernel_read_readvariableop,
(savev2_dense_83_bias_read_readvariableop:
6savev2_lstm_27_lstm_cell_27_kernel_read_readvariableopD
@savev2_lstm_27_lstm_cell_27_recurrent_kernel_read_readvariableop8
4savev2_lstm_27_lstm_cell_27_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop5
1savev2_adam_dense_81_kernel_m_read_readvariableop3
/savev2_adam_dense_81_bias_m_read_readvariableop5
1savev2_adam_dense_82_kernel_m_read_readvariableop3
/savev2_adam_dense_82_bias_m_read_readvariableop5
1savev2_adam_dense_83_kernel_m_read_readvariableop3
/savev2_adam_dense_83_bias_m_read_readvariableopA
=savev2_adam_lstm_27_lstm_cell_27_kernel_m_read_readvariableopK
Gsavev2_adam_lstm_27_lstm_cell_27_recurrent_kernel_m_read_readvariableop?
;savev2_adam_lstm_27_lstm_cell_27_bias_m_read_readvariableop5
1savev2_adam_dense_81_kernel_v_read_readvariableop3
/savev2_adam_dense_81_bias_v_read_readvariableop5
1savev2_adam_dense_82_kernel_v_read_readvariableop3
/savev2_adam_dense_82_bias_v_read_readvariableop5
1savev2_adam_dense_83_kernel_v_read_readvariableop3
/savev2_adam_dense_83_bias_v_read_readvariableopA
=savev2_adam_lstm_27_lstm_cell_27_kernel_v_read_readvariableopK
Gsavev2_adam_lstm_27_lstm_cell_27_recurrent_kernel_v_read_readvariableop?
;savev2_adam_lstm_27_lstm_cell_27_bias_v_read_readvariableop
savev2_const

identity_1ЂMergeV2Checkpointsw
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
_temp/part
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
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:%*
dtype0*Ў
valueЄBЁ%B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHЗ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:%*
dtype0*]
valueTBR%B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B Р
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_81_kernel_read_readvariableop(savev2_dense_81_bias_read_readvariableop*savev2_dense_82_kernel_read_readvariableop(savev2_dense_82_bias_read_readvariableop*savev2_dense_83_kernel_read_readvariableop(savev2_dense_83_bias_read_readvariableop6savev2_lstm_27_lstm_cell_27_kernel_read_readvariableop@savev2_lstm_27_lstm_cell_27_recurrent_kernel_read_readvariableop4savev2_lstm_27_lstm_cell_27_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop1savev2_adam_dense_81_kernel_m_read_readvariableop/savev2_adam_dense_81_bias_m_read_readvariableop1savev2_adam_dense_82_kernel_m_read_readvariableop/savev2_adam_dense_82_bias_m_read_readvariableop1savev2_adam_dense_83_kernel_m_read_readvariableop/savev2_adam_dense_83_bias_m_read_readvariableop=savev2_adam_lstm_27_lstm_cell_27_kernel_m_read_readvariableopGsavev2_adam_lstm_27_lstm_cell_27_recurrent_kernel_m_read_readvariableop;savev2_adam_lstm_27_lstm_cell_27_bias_m_read_readvariableop1savev2_adam_dense_81_kernel_v_read_readvariableop/savev2_adam_dense_81_bias_v_read_readvariableop1savev2_adam_dense_82_kernel_v_read_readvariableop/savev2_adam_dense_82_bias_v_read_readvariableop1savev2_adam_dense_83_kernel_v_read_readvariableop/savev2_adam_dense_83_bias_v_read_readvariableop=savev2_adam_lstm_27_lstm_cell_27_kernel_v_read_readvariableopGsavev2_adam_lstm_27_lstm_cell_27_recurrent_kernel_v_read_readvariableop;savev2_adam_lstm_27_lstm_cell_27_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *3
dtypes)
'2%	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
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

identity_1Identity_1:output:0*
_input_shapes
§: :d2:2:22:2:2::	:	d:: : : : : : : : : :d2:2:22:2:2::	:	d::d2:2:22:2:2::	:	d:: 2(
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
:	:%!

_output_shapes
:	d:!	

_output_shapes	
::
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
:	:%!

_output_shapes
:	d:!

_output_shapes	
::$ 

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
:	:%#!

_output_shapes
:	d:!$

_output_shapes	
::%

_output_shapes
: 
Е
У
while_cond_940084
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_940084___redundant_placeholder04
0while_while_cond_940084___redundant_placeholder14
0while_while_cond_940084___redundant_placeholder24
0while_while_cond_940084___redundant_placeholder3
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
@: : : : :џџџџџџџџџd:џџџџџџџџџd: ::::: 
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
:џџџџџџџџџd:-)
'
_output_shapes
:џџџџџџџџџd:

_output_shapes
: :

_output_shapes
:
Е
У
while_cond_939891
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_939891___redundant_placeholder04
0while_while_cond_939891___redundant_placeholder14
0while_while_cond_939891___redundant_placeholder24
0while_while_cond_939891___redundant_placeholder3
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
@: : : : :џџџџџџџџџd:џџџџџџџџџd: ::::: 
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
:џџџџџџџџџd:-)
'
_output_shapes
:џџџџџџџџџd:

_output_shapes
: :

_output_shapes
:
Ј9
а
while_body_940230
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_27_matmul_readvariableop_resource_0:	H
5while_lstm_cell_27_matmul_1_readvariableop_resource_0:	dC
4while_lstm_cell_27_biasadd_readvariableop_resource_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_27_matmul_readvariableop_resource:	F
3while_lstm_cell_27_matmul_1_readvariableop_resource:	dA
2while_lstm_cell_27_biasadd_readvariableop_resource:	Ђ)while/lstm_cell_27/BiasAdd/ReadVariableOpЂ(while/lstm_cell_27/MatMul/ReadVariableOpЂ*while/lstm_cell_27/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   І
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype0
(while/lstm_cell_27/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_27_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype0К
while/lstm_cell_27/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_27/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЁ
*while/lstm_cell_27/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_27_matmul_1_readvariableop_resource_0*
_output_shapes
:	d*
dtype0Ё
while/lstm_cell_27/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_27/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ
while/lstm_cell_27/addAddV2#while/lstm_cell_27/MatMul:product:0%while/lstm_cell_27/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџ
)while/lstm_cell_27/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_27_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0Ї
while/lstm_cell_27/BiasAddBiasAddwhile/lstm_cell_27/add:z:01while/lstm_cell_27/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџd
"while/lstm_cell_27/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :я
while/lstm_cell_27/splitSplit+while/lstm_cell_27/split/split_dim:output:0#while/lstm_cell_27/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџd:џџџџџџџџџd:џџџџџџџџџd:џџџџџџџџџd*
	num_splitz
while/lstm_cell_27/SigmoidSigmoid!while/lstm_cell_27/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџd|
while/lstm_cell_27/Sigmoid_1Sigmoid!while/lstm_cell_27/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџd
while/lstm_cell_27/mulMul while/lstm_cell_27/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџdt
while/lstm_cell_27/ReluRelu!while/lstm_cell_27/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџd
while/lstm_cell_27/mul_1Mulwhile/lstm_cell_27/Sigmoid:y:0%while/lstm_cell_27/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџd
while/lstm_cell_27/add_1AddV2while/lstm_cell_27/mul:z:0while/lstm_cell_27/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџd|
while/lstm_cell_27/Sigmoid_2Sigmoid!while/lstm_cell_27/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџdq
while/lstm_cell_27/Relu_1Reluwhile/lstm_cell_27/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџd
while/lstm_cell_27/mul_2Mul while/lstm_cell_27/Sigmoid_2:y:0'while/lstm_cell_27/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџdr
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : э
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell_27/mul_2:z:0*
_output_shapes
: *
element_dtype0:щшвM
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
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: y
while/Identity_4Identitywhile/lstm_cell_27/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџdy
while/Identity_5Identitywhile/lstm_cell_27/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџdа

while/NoOpNoOp*^while/lstm_cell_27/BiasAdd/ReadVariableOp)^while/lstm_cell_27/MatMul/ReadVariableOp+^while/lstm_cell_27/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_27_biasadd_readvariableop_resource4while_lstm_cell_27_biasadd_readvariableop_resource_0"l
3while_lstm_cell_27_matmul_1_readvariableop_resource5while_lstm_cell_27_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_27_matmul_readvariableop_resource3while_lstm_cell_27_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџd:џџџџџџџџџd: : : : : 2V
)while/lstm_cell_27/BiasAdd/ReadVariableOp)while/lstm_cell_27/BiasAdd/ReadVariableOp2T
(while/lstm_cell_27/MatMul/ReadVariableOp(while/lstm_cell_27/MatMul/ReadVariableOp2X
*while/lstm_cell_27/MatMul_1/ReadVariableOp*while/lstm_cell_27/MatMul_1/ReadVariableOp: 
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
:џџџџџџџџџd:-)
'
_output_shapes
:џџџџџџџџџd:

_output_shapes
: :

_output_shapes
: 
ы
і
-__inference_lstm_cell_27_layer_call_fn_941863

inputs
states_0
states_1
unknown:	
	unknown_0:	d
	unknown_1:	
identity

identity_1

identity_2ЂStatefulPartitionedCallЈ
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:џџџџџџџџџd:џџџџџџџџџd:џџџџџџџџџd*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_lstm_cell_27_layer_call_and_return_conditional_losses_940025o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџdq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:џџџџџџџџџdq

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:џџџџџџџџџd`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:џџџџџџџџџ:џџџџџџџџџd:џџџџџџџџџd: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:QM
'
_output_shapes
:џџџџџџџџџd
"
_user_specified_name
states/0:QM
'
_output_shapes
:џџџџџџџџџd
"
_user_specified_name
states/1
Е
У
while_cond_941394
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_941394___redundant_placeholder04
0while_while_cond_941394___redundant_placeholder14
0while_while_cond_941394___redundant_placeholder24
0while_while_cond_941394___redundant_placeholder3
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
@: : : : :џџџџџџџџџd:џџџџџџџџџd: ::::: 
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
:џџџџџџџџџd:-)
'
_output_shapes
:џџџџџџџџџd:

_output_shapes
: :

_output_shapes
:
Е
У
while_cond_941249
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_941249___redundant_placeholder04
0while_while_cond_941249___redundant_placeholder14
0while_while_cond_941249___redundant_placeholder24
0while_while_cond_941249___redundant_placeholder3
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
@: : : : :џџџџџџџџџd:џџџџџџџџџd: ::::: 
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
:џџџџџџџџџd:-)
'
_output_shapes
:џџџџџџџџџd:

_output_shapes
: :

_output_shapes
:
Ј9
а
while_body_940498
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_27_matmul_readvariableop_resource_0:	H
5while_lstm_cell_27_matmul_1_readvariableop_resource_0:	dC
4while_lstm_cell_27_biasadd_readvariableop_resource_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_27_matmul_readvariableop_resource:	F
3while_lstm_cell_27_matmul_1_readvariableop_resource:	dA
2while_lstm_cell_27_biasadd_readvariableop_resource:	Ђ)while/lstm_cell_27/BiasAdd/ReadVariableOpЂ(while/lstm_cell_27/MatMul/ReadVariableOpЂ*while/lstm_cell_27/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   І
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype0
(while/lstm_cell_27/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_27_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype0К
while/lstm_cell_27/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_27/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЁ
*while/lstm_cell_27/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_27_matmul_1_readvariableop_resource_0*
_output_shapes
:	d*
dtype0Ё
while/lstm_cell_27/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_27/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ
while/lstm_cell_27/addAddV2#while/lstm_cell_27/MatMul:product:0%while/lstm_cell_27/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџ
)while/lstm_cell_27/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_27_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0Ї
while/lstm_cell_27/BiasAddBiasAddwhile/lstm_cell_27/add:z:01while/lstm_cell_27/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџd
"while/lstm_cell_27/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :я
while/lstm_cell_27/splitSplit+while/lstm_cell_27/split/split_dim:output:0#while/lstm_cell_27/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџd:џџџџџџџџџd:џџџџџџџџџd:џџџџџџџџџd*
	num_splitz
while/lstm_cell_27/SigmoidSigmoid!while/lstm_cell_27/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџd|
while/lstm_cell_27/Sigmoid_1Sigmoid!while/lstm_cell_27/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџd
while/lstm_cell_27/mulMul while/lstm_cell_27/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџdt
while/lstm_cell_27/ReluRelu!while/lstm_cell_27/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџd
while/lstm_cell_27/mul_1Mulwhile/lstm_cell_27/Sigmoid:y:0%while/lstm_cell_27/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџd
while/lstm_cell_27/add_1AddV2while/lstm_cell_27/mul:z:0while/lstm_cell_27/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџd|
while/lstm_cell_27/Sigmoid_2Sigmoid!while/lstm_cell_27/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџdq
while/lstm_cell_27/Relu_1Reluwhile/lstm_cell_27/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџd
while/lstm_cell_27/mul_2Mul while/lstm_cell_27/Sigmoid_2:y:0'while/lstm_cell_27/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџdr
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : э
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell_27/mul_2:z:0*
_output_shapes
: *
element_dtype0:щшвM
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
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: y
while/Identity_4Identitywhile/lstm_cell_27/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџdy
while/Identity_5Identitywhile/lstm_cell_27/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџdа

while/NoOpNoOp*^while/lstm_cell_27/BiasAdd/ReadVariableOp)^while/lstm_cell_27/MatMul/ReadVariableOp+^while/lstm_cell_27/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_27_biasadd_readvariableop_resource4while_lstm_cell_27_biasadd_readvariableop_resource_0"l
3while_lstm_cell_27_matmul_1_readvariableop_resource5while_lstm_cell_27_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_27_matmul_readvariableop_resource3while_lstm_cell_27_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџd:џџџџџџџџџd: : : : : 2V
)while/lstm_cell_27/BiasAdd/ReadVariableOp)while/lstm_cell_27/BiasAdd/ReadVariableOp2T
(while/lstm_cell_27/MatMul/ReadVariableOp(while/lstm_cell_27/MatMul/ReadVariableOp2X
*while/lstm_cell_27/MatMul_1/ReadVariableOp*while/lstm_cell_27/MatMul_1/ReadVariableOp: 
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
:џџџџџџџџџd:-)
'
_output_shapes
:џџџџџџџџџd:

_output_shapes
: :

_output_shapes
: 
m
ћ
I__inference_sequential_27_layer_call_and_return_conditional_losses_941146

inputsF
3lstm_27_lstm_cell_27_matmul_readvariableop_resource:	H
5lstm_27_lstm_cell_27_matmul_1_readvariableop_resource:	dC
4lstm_27_lstm_cell_27_biasadd_readvariableop_resource:	9
'dense_81_matmul_readvariableop_resource:d26
(dense_81_biasadd_readvariableop_resource:29
'dense_82_matmul_readvariableop_resource:226
(dense_82_biasadd_readvariableop_resource:29
'dense_83_matmul_readvariableop_resource:26
(dense_83_biasadd_readvariableop_resource:
identityЂdense_81/BiasAdd/ReadVariableOpЂdense_81/MatMul/ReadVariableOpЂdense_82/BiasAdd/ReadVariableOpЂdense_82/MatMul/ReadVariableOpЂdense_83/BiasAdd/ReadVariableOpЂdense_83/MatMul/ReadVariableOpЂ+lstm_27/lstm_cell_27/BiasAdd/ReadVariableOpЂ*lstm_27/lstm_cell_27/MatMul/ReadVariableOpЂ,lstm_27/lstm_cell_27/MatMul_1/ReadVariableOpЂlstm_27/whileC
lstm_27/ShapeShapeinputs*
T0*
_output_shapes
:e
lstm_27/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
lstm_27/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
lstm_27/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:љ
lstm_27/strided_sliceStridedSlicelstm_27/Shape:output:0$lstm_27/strided_slice/stack:output:0&lstm_27/strided_slice/stack_1:output:0&lstm_27/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
lstm_27/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d
lstm_27/zeros/packedPacklstm_27/strided_slice:output:0lstm_27/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:X
lstm_27/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
lstm_27/zerosFilllstm_27/zeros/packed:output:0lstm_27/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџdZ
lstm_27/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d
lstm_27/zeros_1/packedPacklstm_27/strided_slice:output:0!lstm_27/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:Z
lstm_27/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
lstm_27/zeros_1Filllstm_27/zeros_1/packed:output:0lstm_27/zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџdk
lstm_27/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          }
lstm_27/transpose	Transposeinputslstm_27/transpose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџT
lstm_27/Shape_1Shapelstm_27/transpose:y:0*
T0*
_output_shapes
:g
lstm_27/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: i
lstm_27/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
lstm_27/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
lstm_27/strided_slice_1StridedSlicelstm_27/Shape_1:output:0&lstm_27/strided_slice_1/stack:output:0(lstm_27/strided_slice_1/stack_1:output:0(lstm_27/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskn
#lstm_27/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџЬ
lstm_27/TensorArrayV2TensorListReserve,lstm_27/TensorArrayV2/element_shape:output:0 lstm_27/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
=lstm_27/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   ј
/lstm_27/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_27/transpose:y:0Flstm_27/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвg
lstm_27/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: i
lstm_27/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
lstm_27/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
lstm_27/strided_slice_2StridedSlicelstm_27/transpose:y:0&lstm_27/strided_slice_2/stack:output:0(lstm_27/strided_slice_2/stack_1:output:0(lstm_27/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask
*lstm_27/lstm_cell_27/MatMul/ReadVariableOpReadVariableOp3lstm_27_lstm_cell_27_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0Ў
lstm_27/lstm_cell_27/MatMulMatMul lstm_27/strided_slice_2:output:02lstm_27/lstm_cell_27/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЃ
,lstm_27/lstm_cell_27/MatMul_1/ReadVariableOpReadVariableOp5lstm_27_lstm_cell_27_matmul_1_readvariableop_resource*
_output_shapes
:	d*
dtype0Ј
lstm_27/lstm_cell_27/MatMul_1MatMullstm_27/zeros:output:04lstm_27/lstm_cell_27/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЄ
lstm_27/lstm_cell_27/addAddV2%lstm_27/lstm_cell_27/MatMul:product:0'lstm_27/lstm_cell_27/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџ
+lstm_27/lstm_cell_27/BiasAdd/ReadVariableOpReadVariableOp4lstm_27_lstm_cell_27_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0­
lstm_27/lstm_cell_27/BiasAddBiasAddlstm_27/lstm_cell_27/add:z:03lstm_27/lstm_cell_27/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџf
$lstm_27/lstm_cell_27/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ѕ
lstm_27/lstm_cell_27/splitSplit-lstm_27/lstm_cell_27/split/split_dim:output:0%lstm_27/lstm_cell_27/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџd:џџџџџџџџџd:џџџџџџџџџd:џџџџџџџџџd*
	num_split~
lstm_27/lstm_cell_27/SigmoidSigmoid#lstm_27/lstm_cell_27/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџd
lstm_27/lstm_cell_27/Sigmoid_1Sigmoid#lstm_27/lstm_cell_27/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџd
lstm_27/lstm_cell_27/mulMul"lstm_27/lstm_cell_27/Sigmoid_1:y:0lstm_27/zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџdx
lstm_27/lstm_cell_27/ReluRelu#lstm_27/lstm_cell_27/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџd
lstm_27/lstm_cell_27/mul_1Mul lstm_27/lstm_cell_27/Sigmoid:y:0'lstm_27/lstm_cell_27/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџd
lstm_27/lstm_cell_27/add_1AddV2lstm_27/lstm_cell_27/mul:z:0lstm_27/lstm_cell_27/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџd
lstm_27/lstm_cell_27/Sigmoid_2Sigmoid#lstm_27/lstm_cell_27/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџdu
lstm_27/lstm_cell_27/Relu_1Relulstm_27/lstm_cell_27/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџdЂ
lstm_27/lstm_cell_27/mul_2Mul"lstm_27/lstm_cell_27/Sigmoid_2:y:0)lstm_27/lstm_cell_27/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџdv
%lstm_27/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџd   f
$lstm_27/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :н
lstm_27/TensorArrayV2_1TensorListReserve.lstm_27/TensorArrayV2_1/element_shape:output:0-lstm_27/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвN
lstm_27/timeConst*
_output_shapes
: *
dtype0*
value	B : k
 lstm_27/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ\
lstm_27/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ђ
lstm_27/whileWhile#lstm_27/while/loop_counter:output:0)lstm_27/while/maximum_iterations:output:0lstm_27/time:output:0 lstm_27/TensorArrayV2_1:handle:0lstm_27/zeros:output:0lstm_27/zeros_1:output:0 lstm_27/strided_slice_1:output:0?lstm_27/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_27_lstm_cell_27_matmul_readvariableop_resource5lstm_27_lstm_cell_27_matmul_1_readvariableop_resource4lstm_27_lstm_cell_27_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџd:џџџџџџџџџd: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *%
bodyR
lstm_27_while_body_941041*%
condR
lstm_27_while_cond_941040*K
output_shapes:
8: : : : :џџџџџџџџџd:џџџџџџџџџd: : : : : *
parallel_iterations 
8lstm_27/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџd   ю
*lstm_27/TensorArrayV2Stack/TensorListStackTensorListStacklstm_27/while:output:3Alstm_27/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџd*
element_dtype0*
num_elementsp
lstm_27/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџi
lstm_27/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: i
lstm_27/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Џ
lstm_27/strided_slice_3StridedSlice3lstm_27/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_27/strided_slice_3/stack:output:0(lstm_27/strided_slice_3/stack_1:output:0(lstm_27/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџd*
shrink_axis_maskm
lstm_27/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ў
lstm_27/transpose_1	Transpose3lstm_27/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_27/transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџdc
lstm_27/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    
dense_81/MatMul/ReadVariableOpReadVariableOp'dense_81_matmul_readvariableop_resource*
_output_shapes

:d2*
dtype0
dense_81/MatMulMatMul lstm_27/strided_slice_3:output:0&dense_81/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_81/BiasAdd/ReadVariableOpReadVariableOp(dense_81_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0
dense_81/BiasAddBiasAdddense_81/MatMul:product:0'dense_81/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2b
dense_81/ReluReludense_81/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_82/MatMul/ReadVariableOpReadVariableOp'dense_82_matmul_readvariableop_resource*
_output_shapes

:22*
dtype0
dense_82/MatMulMatMuldense_81/Relu:activations:0&dense_82/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_82/BiasAdd/ReadVariableOpReadVariableOp(dense_82_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0
dense_82/BiasAddBiasAdddense_82/MatMul:product:0'dense_82/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2b
dense_82/ReluReludense_82/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_83/MatMul/ReadVariableOpReadVariableOp'dense_83_matmul_readvariableop_resource*
_output_shapes

:2*
dtype0
dense_83/MatMulMatMuldense_82/Relu:activations:0&dense_83/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
dense_83/BiasAdd/ReadVariableOpReadVariableOp(dense_83_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_83/BiasAddBiasAdddense_83/MatMul:product:0'dense_83/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџh
IdentityIdentitydense_83/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџЉ
NoOpNoOp ^dense_81/BiasAdd/ReadVariableOp^dense_81/MatMul/ReadVariableOp ^dense_82/BiasAdd/ReadVariableOp^dense_82/MatMul/ReadVariableOp ^dense_83/BiasAdd/ReadVariableOp^dense_83/MatMul/ReadVariableOp,^lstm_27/lstm_cell_27/BiasAdd/ReadVariableOp+^lstm_27/lstm_cell_27/MatMul/ReadVariableOp-^lstm_27/lstm_cell_27/MatMul_1/ReadVariableOp^lstm_27/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):џџџџџџџџџ: : : : : : : : : 2B
dense_81/BiasAdd/ReadVariableOpdense_81/BiasAdd/ReadVariableOp2@
dense_81/MatMul/ReadVariableOpdense_81/MatMul/ReadVariableOp2B
dense_82/BiasAdd/ReadVariableOpdense_82/BiasAdd/ReadVariableOp2@
dense_82/MatMul/ReadVariableOpdense_82/MatMul/ReadVariableOp2B
dense_83/BiasAdd/ReadVariableOpdense_83/BiasAdd/ReadVariableOp2@
dense_83/MatMul/ReadVariableOpdense_83/MatMul/ReadVariableOp2Z
+lstm_27/lstm_cell_27/BiasAdd/ReadVariableOp+lstm_27/lstm_cell_27/BiasAdd/ReadVariableOp2X
*lstm_27/lstm_cell_27/MatMul/ReadVariableOp*lstm_27/lstm_cell_27/MatMul/ReadVariableOp2\
,lstm_27/lstm_cell_27/MatMul_1/ReadVariableOp,lstm_27/lstm_cell_27/MatMul_1/ReadVariableOp2
lstm_27/whilelstm_27/while:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
џ	
н
.__inference_sequential_27_layer_call_fn_940816

inputs
unknown:	
	unknown_0:	d
	unknown_1:	
	unknown_2:d2
	unknown_3:2
	unknown_4:22
	unknown_5:2
	unknown_6:2
	unknown_7:
identityЂStatefulPartitionedCallЙ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*+
_read_only_resource_inputs
		*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_27_layer_call_and_return_conditional_losses_940643o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):џџџџџџџџџ: : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
щB
а

lstm_27_while_body_940876,
(lstm_27_while_lstm_27_while_loop_counter2
.lstm_27_while_lstm_27_while_maximum_iterations
lstm_27_while_placeholder
lstm_27_while_placeholder_1
lstm_27_while_placeholder_2
lstm_27_while_placeholder_3+
'lstm_27_while_lstm_27_strided_slice_1_0g
clstm_27_while_tensorarrayv2read_tensorlistgetitem_lstm_27_tensorarrayunstack_tensorlistfromtensor_0N
;lstm_27_while_lstm_cell_27_matmul_readvariableop_resource_0:	P
=lstm_27_while_lstm_cell_27_matmul_1_readvariableop_resource_0:	dK
<lstm_27_while_lstm_cell_27_biasadd_readvariableop_resource_0:	
lstm_27_while_identity
lstm_27_while_identity_1
lstm_27_while_identity_2
lstm_27_while_identity_3
lstm_27_while_identity_4
lstm_27_while_identity_5)
%lstm_27_while_lstm_27_strided_slice_1e
alstm_27_while_tensorarrayv2read_tensorlistgetitem_lstm_27_tensorarrayunstack_tensorlistfromtensorL
9lstm_27_while_lstm_cell_27_matmul_readvariableop_resource:	N
;lstm_27_while_lstm_cell_27_matmul_1_readvariableop_resource:	dI
:lstm_27_while_lstm_cell_27_biasadd_readvariableop_resource:	Ђ1lstm_27/while/lstm_cell_27/BiasAdd/ReadVariableOpЂ0lstm_27/while/lstm_cell_27/MatMul/ReadVariableOpЂ2lstm_27/while/lstm_cell_27/MatMul_1/ReadVariableOp
?lstm_27/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   Ю
1lstm_27/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_27_while_tensorarrayv2read_tensorlistgetitem_lstm_27_tensorarrayunstack_tensorlistfromtensor_0lstm_27_while_placeholderHlstm_27/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype0­
0lstm_27/while/lstm_cell_27/MatMul/ReadVariableOpReadVariableOp;lstm_27_while_lstm_cell_27_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype0в
!lstm_27/while/lstm_cell_27/MatMulMatMul8lstm_27/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_27/while/lstm_cell_27/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџБ
2lstm_27/while/lstm_cell_27/MatMul_1/ReadVariableOpReadVariableOp=lstm_27_while_lstm_cell_27_matmul_1_readvariableop_resource_0*
_output_shapes
:	d*
dtype0Й
#lstm_27/while/lstm_cell_27/MatMul_1MatMullstm_27_while_placeholder_2:lstm_27/while/lstm_cell_27/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЖ
lstm_27/while/lstm_cell_27/addAddV2+lstm_27/while/lstm_cell_27/MatMul:product:0-lstm_27/while/lstm_cell_27/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџЋ
1lstm_27/while/lstm_cell_27/BiasAdd/ReadVariableOpReadVariableOp<lstm_27_while_lstm_cell_27_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0П
"lstm_27/while/lstm_cell_27/BiasAddBiasAdd"lstm_27/while/lstm_cell_27/add:z:09lstm_27/while/lstm_cell_27/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџl
*lstm_27/while/lstm_cell_27/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
 lstm_27/while/lstm_cell_27/splitSplit3lstm_27/while/lstm_cell_27/split/split_dim:output:0+lstm_27/while/lstm_cell_27/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџd:џџџџџџџџџd:џџџџџџџџџd:џџџџџџџџџd*
	num_split
"lstm_27/while/lstm_cell_27/SigmoidSigmoid)lstm_27/while/lstm_cell_27/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџd
$lstm_27/while/lstm_cell_27/Sigmoid_1Sigmoid)lstm_27/while/lstm_cell_27/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџd
lstm_27/while/lstm_cell_27/mulMul(lstm_27/while/lstm_cell_27/Sigmoid_1:y:0lstm_27_while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџd
lstm_27/while/lstm_cell_27/ReluRelu)lstm_27/while/lstm_cell_27/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџdА
 lstm_27/while/lstm_cell_27/mul_1Mul&lstm_27/while/lstm_cell_27/Sigmoid:y:0-lstm_27/while/lstm_cell_27/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџdЅ
 lstm_27/while/lstm_cell_27/add_1AddV2"lstm_27/while/lstm_cell_27/mul:z:0$lstm_27/while/lstm_cell_27/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџd
$lstm_27/while/lstm_cell_27/Sigmoid_2Sigmoid)lstm_27/while/lstm_cell_27/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџd
!lstm_27/while/lstm_cell_27/Relu_1Relu$lstm_27/while/lstm_cell_27/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџdД
 lstm_27/while/lstm_cell_27/mul_2Mul(lstm_27/while/lstm_cell_27/Sigmoid_2:y:0/lstm_27/while/lstm_cell_27/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџdz
8lstm_27/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : 
2lstm_27/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_27_while_placeholder_1Alstm_27/while/TensorArrayV2Write/TensorListSetItem/index:output:0$lstm_27/while/lstm_cell_27/mul_2:z:0*
_output_shapes
: *
element_dtype0:щшвU
lstm_27/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :t
lstm_27/while/addAddV2lstm_27_while_placeholderlstm_27/while/add/y:output:0*
T0*
_output_shapes
: W
lstm_27/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
lstm_27/while/add_1AddV2(lstm_27_while_lstm_27_while_loop_counterlstm_27/while/add_1/y:output:0*
T0*
_output_shapes
: q
lstm_27/while/IdentityIdentitylstm_27/while/add_1:z:0^lstm_27/while/NoOp*
T0*
_output_shapes
: 
lstm_27/while/Identity_1Identity.lstm_27_while_lstm_27_while_maximum_iterations^lstm_27/while/NoOp*
T0*
_output_shapes
: q
lstm_27/while/Identity_2Identitylstm_27/while/add:z:0^lstm_27/while/NoOp*
T0*
_output_shapes
: 
lstm_27/while/Identity_3IdentityBlstm_27/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_27/while/NoOp*
T0*
_output_shapes
: 
lstm_27/while/Identity_4Identity$lstm_27/while/lstm_cell_27/mul_2:z:0^lstm_27/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџd
lstm_27/while/Identity_5Identity$lstm_27/while/lstm_cell_27/add_1:z:0^lstm_27/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџd№
lstm_27/while/NoOpNoOp2^lstm_27/while/lstm_cell_27/BiasAdd/ReadVariableOp1^lstm_27/while/lstm_cell_27/MatMul/ReadVariableOp3^lstm_27/while/lstm_cell_27/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "9
lstm_27_while_identitylstm_27/while/Identity:output:0"=
lstm_27_while_identity_1!lstm_27/while/Identity_1:output:0"=
lstm_27_while_identity_2!lstm_27/while/Identity_2:output:0"=
lstm_27_while_identity_3!lstm_27/while/Identity_3:output:0"=
lstm_27_while_identity_4!lstm_27/while/Identity_4:output:0"=
lstm_27_while_identity_5!lstm_27/while/Identity_5:output:0"P
%lstm_27_while_lstm_27_strided_slice_1'lstm_27_while_lstm_27_strided_slice_1_0"z
:lstm_27_while_lstm_cell_27_biasadd_readvariableop_resource<lstm_27_while_lstm_cell_27_biasadd_readvariableop_resource_0"|
;lstm_27_while_lstm_cell_27_matmul_1_readvariableop_resource=lstm_27_while_lstm_cell_27_matmul_1_readvariableop_resource_0"x
9lstm_27_while_lstm_cell_27_matmul_readvariableop_resource;lstm_27_while_lstm_cell_27_matmul_readvariableop_resource_0"Ш
alstm_27_while_tensorarrayv2read_tensorlistgetitem_lstm_27_tensorarrayunstack_tensorlistfromtensorclstm_27_while_tensorarrayv2read_tensorlistgetitem_lstm_27_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџd:џџџџџџџџџd: : : : : 2f
1lstm_27/while/lstm_cell_27/BiasAdd/ReadVariableOp1lstm_27/while/lstm_cell_27/BiasAdd/ReadVariableOp2d
0lstm_27/while/lstm_cell_27/MatMul/ReadVariableOp0lstm_27/while/lstm_cell_27/MatMul/ReadVariableOp2h
2lstm_27/while/lstm_cell_27/MatMul_1/ReadVariableOp2lstm_27/while/lstm_cell_27/MatMul_1/ReadVariableOp: 
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
:џџџџџџџџџd:-)
'
_output_shapes
:џџџџџџџџџd:

_output_shapes
: :

_output_shapes
: 
е

H__inference_lstm_cell_27_layer_call_and_return_conditional_losses_940025

inputs

states
states_11
matmul_readvariableop_resource:	3
 matmul_1_readvariableop_resource:	d.
biasadd_readvariableop_resource:	
identity

identity_1

identity_2ЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpЂMatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџy
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	d*
dtype0n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ж
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџd:џџџџџџџџџd:џџџџџџџџџd:џџџџџџџџџd*
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:џџџџџџџџџdV
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:џџџџџџџџџdU
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:џџџџџџџџџdN
ReluRelusplit:output:2*
T0*'
_output_shapes
:џџџџџџџџџd_
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџdT
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџdV
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:џџџџџџџџџdK
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџdc
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџdX
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџdZ

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџdZ

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџd
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:џџџџџџџџџ:џџџџџџџџџd:џџџџџџџџџd: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџd
 
_user_specified_namestates:OK
'
_output_shapes
:џџџџџџџџџd
 
_user_specified_namestates
Ч	
ѕ
D__inference_dense_83_layer_call_and_return_conditional_losses_941829

inputs0
matmul_readvariableop_resource:2-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:2*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ2: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ2
 
_user_specified_nameinputs
$
у
while_body_939892
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0.
while_lstm_cell_27_939916_0:	.
while_lstm_cell_27_939918_0:	d*
while_lstm_cell_27_939920_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor,
while_lstm_cell_27_939916:	,
while_lstm_cell_27_939918:	d(
while_lstm_cell_27_939920:	Ђ*while/lstm_cell_27/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   І
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype0Г
*while/lstm_cell_27/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_27_939916_0while_lstm_cell_27_939918_0while_lstm_cell_27_939920_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:џџџџџџџџџd:џџџџџџџџџd:џџџџџџџџџd*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_lstm_cell_27_layer_call_and_return_conditional_losses_939877r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : 
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:03while/lstm_cell_27/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:щшвM
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
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_4Identity3while/lstm_cell_27/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџd
while/Identity_5Identity3while/lstm_cell_27/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџdy

while/NoOpNoOp+^while/lstm_cell_27/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"8
while_lstm_cell_27_939916while_lstm_cell_27_939916_0"8
while_lstm_cell_27_939918while_lstm_cell_27_939918_0"8
while_lstm_cell_27_939920while_lstm_cell_27_939920_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџd:џџџџџџџџџd: : : : : 2X
*while/lstm_cell_27/StatefulPartitionedCall*while/lstm_cell_27/StatefulPartitionedCall: 
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
:џџџџџџџџџd:-)
'
_output_shapes
:џџџџџџџџџd:

_output_shapes
: :

_output_shapes
: 
ЬK

C__inference_lstm_27_layer_call_and_return_conditional_losses_941335
inputs_0>
+lstm_cell_27_matmul_readvariableop_resource:	@
-lstm_cell_27_matmul_1_readvariableop_resource:	d;
,lstm_cell_27_biasadd_readvariableop_resource:	
identityЂ#lstm_cell_27/BiasAdd/ReadVariableOpЂ"lstm_cell_27/MatMul/ReadVariableOpЂ$lstm_cell_27/MatMul_1/ReadVariableOpЂwhile=
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
valueB:б
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
:џџџџџџџџџdR
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
:џџџџџџџџџdc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџD
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
valueB:л
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
џџџџџџџџџД
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв_
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
valueB:щ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask
"lstm_cell_27/MatMul/ReadVariableOpReadVariableOp+lstm_cell_27_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
lstm_cell_27/MatMulMatMulstrided_slice_2:output:0*lstm_cell_27/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ
$lstm_cell_27/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_27_matmul_1_readvariableop_resource*
_output_shapes
:	d*
dtype0
lstm_cell_27/MatMul_1MatMulzeros:output:0,lstm_cell_27/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ
lstm_cell_27/addAddV2lstm_cell_27/MatMul:product:0lstm_cell_27/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџ
#lstm_cell_27/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_27_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
lstm_cell_27/BiasAddBiasAddlstm_cell_27/add:z:0+lstm_cell_27/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ^
lstm_cell_27/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :н
lstm_cell_27/splitSplit%lstm_cell_27/split/split_dim:output:0lstm_cell_27/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџd:џџџџџџџџџd:џџџџџџџџџd:џџџџџџџџџd*
	num_splitn
lstm_cell_27/SigmoidSigmoidlstm_cell_27/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџdp
lstm_cell_27/Sigmoid_1Sigmoidlstm_cell_27/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџdw
lstm_cell_27/mulMullstm_cell_27/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџdh
lstm_cell_27/ReluRelulstm_cell_27/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџd
lstm_cell_27/mul_1Mullstm_cell_27/Sigmoid:y:0lstm_cell_27/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџd{
lstm_cell_27/add_1AddV2lstm_cell_27/mul:z:0lstm_cell_27/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџdp
lstm_cell_27/Sigmoid_2Sigmoidlstm_cell_27/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџde
lstm_cell_27/Relu_1Relulstm_cell_27/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџd
lstm_cell_27/mul_2Mullstm_cell_27/Sigmoid_2:y:0!lstm_cell_27/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџdn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџd   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Х
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвF
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
џџџџџџџџџT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_27_matmul_readvariableop_resource-lstm_cell_27_matmul_1_readvariableop_resource,lstm_cell_27_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџd:џџџџџџџџџd: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_941250*
condR
while_cond_941249*K
output_shapes:
8: : : : :џџџџџџџџџd:џџџџџџџџџd: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџd   ж
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџd*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџd*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџd[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџdР
NoOpNoOp$^lstm_cell_27/BiasAdd/ReadVariableOp#^lstm_cell_27/MatMul/ReadVariableOp%^lstm_cell_27/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџџџџџџџџџџ: : : 2J
#lstm_cell_27/BiasAdd/ReadVariableOp#lstm_cell_27/BiasAdd/ReadVariableOp2H
"lstm_cell_27/MatMul/ReadVariableOp"lstm_cell_27/MatMul/ReadVariableOp2L
$lstm_cell_27/MatMul_1/ReadVariableOp$lstm_cell_27/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/0


ф
.__inference_sequential_27_layer_call_fn_940395
lstm_27_input
unknown:	
	unknown_0:	d
	unknown_1:	
	unknown_2:d2
	unknown_3:2
	unknown_4:22
	unknown_5:2
	unknown_6:2
	unknown_7:
identityЂStatefulPartitionedCallР
StatefulPartitionedCallStatefulPartitionedCalllstm_27_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*+
_read_only_resource_inputs
		*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_27_layer_call_and_return_conditional_losses_940374o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):џџџџџџџџџ: : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
+
_output_shapes
:џџџџџџџџџ
'
_user_specified_namelstm_27_input
ЉK

C__inference_lstm_27_layer_call_and_return_conditional_losses_941625

inputs>
+lstm_cell_27_matmul_readvariableop_resource:	@
-lstm_cell_27_matmul_1_readvariableop_resource:	d;
,lstm_cell_27_biasadd_readvariableop_resource:	
identityЂ#lstm_cell_27/BiasAdd/ReadVariableOpЂ"lstm_cell_27/MatMul/ReadVariableOpЂ$lstm_cell_27/MatMul_1/ReadVariableOpЂwhile;
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
valueB:б
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
:џџџџџџџџџdR
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
:џџџџџџџџџdc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџD
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
valueB:л
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
џџџџџџџџџД
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв_
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
valueB:щ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask
"lstm_cell_27/MatMul/ReadVariableOpReadVariableOp+lstm_cell_27_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
lstm_cell_27/MatMulMatMulstrided_slice_2:output:0*lstm_cell_27/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ
$lstm_cell_27/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_27_matmul_1_readvariableop_resource*
_output_shapes
:	d*
dtype0
lstm_cell_27/MatMul_1MatMulzeros:output:0,lstm_cell_27/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ
lstm_cell_27/addAddV2lstm_cell_27/MatMul:product:0lstm_cell_27/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџ
#lstm_cell_27/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_27_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
lstm_cell_27/BiasAddBiasAddlstm_cell_27/add:z:0+lstm_cell_27/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ^
lstm_cell_27/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :н
lstm_cell_27/splitSplit%lstm_cell_27/split/split_dim:output:0lstm_cell_27/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџd:џџџџџџџџџd:џџџџџџџџџd:џџџџџџџџџd*
	num_splitn
lstm_cell_27/SigmoidSigmoidlstm_cell_27/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџdp
lstm_cell_27/Sigmoid_1Sigmoidlstm_cell_27/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџdw
lstm_cell_27/mulMullstm_cell_27/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџdh
lstm_cell_27/ReluRelulstm_cell_27/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџd
lstm_cell_27/mul_1Mullstm_cell_27/Sigmoid:y:0lstm_cell_27/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџd{
lstm_cell_27/add_1AddV2lstm_cell_27/mul:z:0lstm_cell_27/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџdp
lstm_cell_27/Sigmoid_2Sigmoidlstm_cell_27/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџde
lstm_cell_27/Relu_1Relulstm_cell_27/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџd
lstm_cell_27/mul_2Mullstm_cell_27/Sigmoid_2:y:0!lstm_cell_27/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџdn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџd   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Х
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвF
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
џџџџџџџџџT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_27_matmul_readvariableop_resource-lstm_cell_27_matmul_1_readvariableop_resource,lstm_cell_27_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџd:џџџџџџџџџd: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_941540*
condR
while_cond_941539*K
output_shapes:
8: : : : :џџџџџџџџџd:џџџџџџџџџd: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџd   ж
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџd*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџd*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџd[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџdР
NoOpNoOp$^lstm_cell_27/BiasAdd/ReadVariableOp#^lstm_cell_27/MatMul/ReadVariableOp%^lstm_cell_27/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:џџџџџџџџџ: : : 2J
#lstm_cell_27/BiasAdd/ReadVariableOp#lstm_cell_27/BiasAdd/ReadVariableOp2H
"lstm_cell_27/MatMul/ReadVariableOp"lstm_cell_27/MatMul/ReadVariableOp2L
$lstm_cell_27/MatMul_1/ReadVariableOp$lstm_cell_27/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
џ	
н
.__inference_sequential_27_layer_call_fn_940793

inputs
unknown:	
	unknown_0:	d
	unknown_1:	
	unknown_2:d2
	unknown_3:2
	unknown_4:22
	unknown_5:2
	unknown_6:2
	unknown_7:
identityЂStatefulPartitionedCallЙ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*+
_read_only_resource_inputs
		*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_27_layer_call_and_return_conditional_losses_940374o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):џџџџџџџџџ: : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ј9
а
while_body_941685
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_27_matmul_readvariableop_resource_0:	H
5while_lstm_cell_27_matmul_1_readvariableop_resource_0:	dC
4while_lstm_cell_27_biasadd_readvariableop_resource_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_27_matmul_readvariableop_resource:	F
3while_lstm_cell_27_matmul_1_readvariableop_resource:	dA
2while_lstm_cell_27_biasadd_readvariableop_resource:	Ђ)while/lstm_cell_27/BiasAdd/ReadVariableOpЂ(while/lstm_cell_27/MatMul/ReadVariableOpЂ*while/lstm_cell_27/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   І
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype0
(while/lstm_cell_27/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_27_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype0К
while/lstm_cell_27/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_27/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЁ
*while/lstm_cell_27/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_27_matmul_1_readvariableop_resource_0*
_output_shapes
:	d*
dtype0Ё
while/lstm_cell_27/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_27/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ
while/lstm_cell_27/addAddV2#while/lstm_cell_27/MatMul:product:0%while/lstm_cell_27/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџ
)while/lstm_cell_27/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_27_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0Ї
while/lstm_cell_27/BiasAddBiasAddwhile/lstm_cell_27/add:z:01while/lstm_cell_27/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџd
"while/lstm_cell_27/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :я
while/lstm_cell_27/splitSplit+while/lstm_cell_27/split/split_dim:output:0#while/lstm_cell_27/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџd:џџџџџџџџџd:џџџџџџџџџd:џџџџџџџџџd*
	num_splitz
while/lstm_cell_27/SigmoidSigmoid!while/lstm_cell_27/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџd|
while/lstm_cell_27/Sigmoid_1Sigmoid!while/lstm_cell_27/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџd
while/lstm_cell_27/mulMul while/lstm_cell_27/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџdt
while/lstm_cell_27/ReluRelu!while/lstm_cell_27/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџd
while/lstm_cell_27/mul_1Mulwhile/lstm_cell_27/Sigmoid:y:0%while/lstm_cell_27/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџd
while/lstm_cell_27/add_1AddV2while/lstm_cell_27/mul:z:0while/lstm_cell_27/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџd|
while/lstm_cell_27/Sigmoid_2Sigmoid!while/lstm_cell_27/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџdq
while/lstm_cell_27/Relu_1Reluwhile/lstm_cell_27/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџd
while/lstm_cell_27/mul_2Mul while/lstm_cell_27/Sigmoid_2:y:0'while/lstm_cell_27/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџdr
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : э
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell_27/mul_2:z:0*
_output_shapes
: *
element_dtype0:щшвM
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
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: y
while/Identity_4Identitywhile/lstm_cell_27/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџdy
while/Identity_5Identitywhile/lstm_cell_27/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџdа

while/NoOpNoOp*^while/lstm_cell_27/BiasAdd/ReadVariableOp)^while/lstm_cell_27/MatMul/ReadVariableOp+^while/lstm_cell_27/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_27_biasadd_readvariableop_resource4while_lstm_cell_27_biasadd_readvariableop_resource_0"l
3while_lstm_cell_27_matmul_1_readvariableop_resource5while_lstm_cell_27_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_27_matmul_readvariableop_resource3while_lstm_cell_27_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџd:џџџџџџџџџd: : : : : 2V
)while/lstm_cell_27/BiasAdd/ReadVariableOp)while/lstm_cell_27/BiasAdd/ReadVariableOp2T
(while/lstm_cell_27/MatMul/ReadVariableOp(while/lstm_cell_27/MatMul/ReadVariableOp2X
*while/lstm_cell_27/MatMul_1/ReadVariableOp*while/lstm_cell_27/MatMul_1/ReadVariableOp: 
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
:џџџџџџџџџd:-)
'
_output_shapes
:џџџџџџџџџd:

_output_shapes
: :

_output_shapes
: 
Ј9
а
while_body_941395
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_27_matmul_readvariableop_resource_0:	H
5while_lstm_cell_27_matmul_1_readvariableop_resource_0:	dC
4while_lstm_cell_27_biasadd_readvariableop_resource_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_27_matmul_readvariableop_resource:	F
3while_lstm_cell_27_matmul_1_readvariableop_resource:	dA
2while_lstm_cell_27_biasadd_readvariableop_resource:	Ђ)while/lstm_cell_27/BiasAdd/ReadVariableOpЂ(while/lstm_cell_27/MatMul/ReadVariableOpЂ*while/lstm_cell_27/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   І
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype0
(while/lstm_cell_27/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_27_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype0К
while/lstm_cell_27/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_27/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЁ
*while/lstm_cell_27/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_27_matmul_1_readvariableop_resource_0*
_output_shapes
:	d*
dtype0Ё
while/lstm_cell_27/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_27/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ
while/lstm_cell_27/addAddV2#while/lstm_cell_27/MatMul:product:0%while/lstm_cell_27/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџ
)while/lstm_cell_27/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_27_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0Ї
while/lstm_cell_27/BiasAddBiasAddwhile/lstm_cell_27/add:z:01while/lstm_cell_27/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџd
"while/lstm_cell_27/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :я
while/lstm_cell_27/splitSplit+while/lstm_cell_27/split/split_dim:output:0#while/lstm_cell_27/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџd:џџџџџџџџџd:џџџџџџџџџd:џџџџџџџџџd*
	num_splitz
while/lstm_cell_27/SigmoidSigmoid!while/lstm_cell_27/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџd|
while/lstm_cell_27/Sigmoid_1Sigmoid!while/lstm_cell_27/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџd
while/lstm_cell_27/mulMul while/lstm_cell_27/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџdt
while/lstm_cell_27/ReluRelu!while/lstm_cell_27/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџd
while/lstm_cell_27/mul_1Mulwhile/lstm_cell_27/Sigmoid:y:0%while/lstm_cell_27/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџd
while/lstm_cell_27/add_1AddV2while/lstm_cell_27/mul:z:0while/lstm_cell_27/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџd|
while/lstm_cell_27/Sigmoid_2Sigmoid!while/lstm_cell_27/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџdq
while/lstm_cell_27/Relu_1Reluwhile/lstm_cell_27/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџd
while/lstm_cell_27/mul_2Mul while/lstm_cell_27/Sigmoid_2:y:0'while/lstm_cell_27/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџdr
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : э
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell_27/mul_2:z:0*
_output_shapes
: *
element_dtype0:щшвM
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
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: y
while/Identity_4Identitywhile/lstm_cell_27/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџdy
while/Identity_5Identitywhile/lstm_cell_27/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџdа

while/NoOpNoOp*^while/lstm_cell_27/BiasAdd/ReadVariableOp)^while/lstm_cell_27/MatMul/ReadVariableOp+^while/lstm_cell_27/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_27_biasadd_readvariableop_resource4while_lstm_cell_27_biasadd_readvariableop_resource_0"l
3while_lstm_cell_27_matmul_1_readvariableop_resource5while_lstm_cell_27_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_27_matmul_readvariableop_resource3while_lstm_cell_27_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџd:џџџџџџџџџd: : : : : 2V
)while/lstm_cell_27/BiasAdd/ReadVariableOp)while/lstm_cell_27/BiasAdd/ReadVariableOp2T
(while/lstm_cell_27/MatMul/ReadVariableOp(while/lstm_cell_27/MatMul/ReadVariableOp2X
*while/lstm_cell_27/MatMul_1/ReadVariableOp*while/lstm_cell_27/MatMul_1/ReadVariableOp: 
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
:џџџџџџџџџd:-)
'
_output_shapes
:џџџџџџџџџd:

_output_shapes
: :

_output_shapes
: "Е	L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Л
serving_defaultЇ
K
lstm_27_input:
serving_default_lstm_27_input:0џџџџџџџџџ<
dense_830
StatefulPartitionedCall:0џџџџџџџџџtensorflow/serving/predict:гР

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
к
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
Л
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
Л
	variables
 trainable_variables
!regularization_losses
"	keras_api
#__call__
*$&call_and_return_all_conditional_losses

%kernel
&bias"
_tf_keras_layer
Л
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
Ъ
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
э
7trace_0
8trace_1
9trace_2
:trace_32
.__inference_sequential_27_layer_call_fn_940395
.__inference_sequential_27_layer_call_fn_940793
.__inference_sequential_27_layer_call_fn_940816
.__inference_sequential_27_layer_call_fn_940687П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 z7trace_0z8trace_1z9trace_2z:trace_3
й
;trace_0
<trace_1
=trace_2
>trace_32ю
I__inference_sequential_27_layer_call_and_return_conditional_losses_940981
I__inference_sequential_27_layer_call_and_return_conditional_losses_941146
I__inference_sequential_27_layer_call_and_return_conditional_losses_940713
I__inference_sequential_27_layer_call_and_return_conditional_losses_940739П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 z;trace_0z<trace_1z=trace_2z>trace_3
вBЯ
!__inference__wrapped_model_939810lstm_27_input"
В
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 

?iter

@beta_1

Abeta_2
	Bdecay
Clearning_ratemm%m&m-m.m/m0m1mvv%v&v-v.v/v0v1v"
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
Й

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
ъ
Ktrace_0
Ltrace_1
Mtrace_2
Ntrace_32џ
(__inference_lstm_27_layer_call_fn_941157
(__inference_lstm_27_layer_call_fn_941168
(__inference_lstm_27_layer_call_fn_941179
(__inference_lstm_27_layer_call_fn_941190д
ЫВЧ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zKtrace_0zLtrace_1zMtrace_2zNtrace_3
ж
Otrace_0
Ptrace_1
Qtrace_2
Rtrace_32ы
C__inference_lstm_27_layer_call_and_return_conditional_losses_941335
C__inference_lstm_27_layer_call_and_return_conditional_losses_941480
C__inference_lstm_27_layer_call_and_return_conditional_losses_941625
C__inference_lstm_27_layer_call_and_return_conditional_losses_941770д
ЫВЧ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zOtrace_0zPtrace_1zQtrace_2zRtrace_3
"
_generic_user_object
ј
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
­
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
э
`trace_02а
)__inference_dense_81_layer_call_fn_941779Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 z`trace_0

atrace_02ы
D__inference_dense_81_layer_call_and_return_conditional_losses_941790Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zatrace_0
!:d22dense_81/kernel
:22dense_81/bias
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
­
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
э
gtrace_02а
)__inference_dense_82_layer_call_fn_941799Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zgtrace_0

htrace_02ы
D__inference_dense_82_layer_call_and_return_conditional_losses_941810Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zhtrace_0
!:222dense_82/kernel
:22dense_82/bias
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
­
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
э
ntrace_02а
)__inference_dense_83_layer_call_fn_941819Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zntrace_0

otrace_02ы
D__inference_dense_83_layer_call_and_return_conditional_losses_941829Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zotrace_0
!:22dense_83/kernel
:2dense_83/bias
.:,	2lstm_27/lstm_cell_27/kernel
8:6	d2%lstm_27/lstm_cell_27/recurrent_kernel
(:&2lstm_27/lstm_cell_27/bias
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
B
.__inference_sequential_27_layer_call_fn_940395lstm_27_input"П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
џBќ
.__inference_sequential_27_layer_call_fn_940793inputs"П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
џBќ
.__inference_sequential_27_layer_call_fn_940816inputs"П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
.__inference_sequential_27_layer_call_fn_940687lstm_27_input"П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
I__inference_sequential_27_layer_call_and_return_conditional_losses_940981inputs"П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
I__inference_sequential_27_layer_call_and_return_conditional_losses_941146inputs"П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ЁB
I__inference_sequential_27_layer_call_and_return_conditional_losses_940713lstm_27_input"П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ЁB
I__inference_sequential_27_layer_call_and_return_conditional_losses_940739lstm_27_input"П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
бBЮ
$__inference_signature_wrapper_940770lstm_27_input"
В
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
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
B
(__inference_lstm_27_layer_call_fn_941157inputs/0"д
ЫВЧ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
(__inference_lstm_27_layer_call_fn_941168inputs/0"д
ЫВЧ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
(__inference_lstm_27_layer_call_fn_941179inputs"д
ЫВЧ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
(__inference_lstm_27_layer_call_fn_941190inputs"д
ЫВЧ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ЋBЈ
C__inference_lstm_27_layer_call_and_return_conditional_losses_941335inputs/0"д
ЫВЧ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ЋBЈ
C__inference_lstm_27_layer_call_and_return_conditional_losses_941480inputs/0"д
ЫВЧ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ЉBІ
C__inference_lstm_27_layer_call_and_return_conditional_losses_941625inputs"д
ЫВЧ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ЉBІ
C__inference_lstm_27_layer_call_and_return_conditional_losses_941770inputs"д
ЫВЧ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
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
­
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
е
wtrace_0
xtrace_12
-__inference_lstm_cell_27_layer_call_fn_941846
-__inference_lstm_cell_27_layer_call_fn_941863Н
ДВА
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zwtrace_0zxtrace_1

ytrace_0
ztrace_12д
H__inference_lstm_cell_27_layer_call_and_return_conditional_losses_941895
H__inference_lstm_cell_27_layer_call_and_return_conditional_losses_941927Н
ДВА
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
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
нBк
)__inference_dense_81_layer_call_fn_941779inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
јBѕ
D__inference_dense_81_layer_call_and_return_conditional_losses_941790inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
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
нBк
)__inference_dense_82_layer_call_fn_941799inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
јBѕ
D__inference_dense_82_layer_call_and_return_conditional_losses_941810inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
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
нBк
)__inference_dense_83_layer_call_fn_941819inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
јBѕ
D__inference_dense_83_layer_call_and_return_conditional_losses_941829inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
N
{	variables
|	keras_api
	}total
	~count"
_tf_keras_metric
b
	variables
	keras_api

total

count

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
B
-__inference_lstm_cell_27_layer_call_fn_941846inputsstates/0states/1"Н
ДВА
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
-__inference_lstm_cell_27_layer_call_fn_941863inputsstates/0states/1"Н
ДВА
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ЋBЈ
H__inference_lstm_cell_27_layer_call_and_return_conditional_losses_941895inputsstates/0states/1"Н
ДВА
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ЋBЈ
H__inference_lstm_cell_27_layer_call_and_return_conditional_losses_941927inputsstates/0states/1"Н
ДВА
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
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
0
1"
trackable_list_wrapper
-
	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
&:$d22Adam/dense_81/kernel/m
 :22Adam/dense_81/bias/m
&:$222Adam/dense_82/kernel/m
 :22Adam/dense_82/bias/m
&:$22Adam/dense_83/kernel/m
 :2Adam/dense_83/bias/m
3:1	2"Adam/lstm_27/lstm_cell_27/kernel/m
=:;	d2,Adam/lstm_27/lstm_cell_27/recurrent_kernel/m
-:+2 Adam/lstm_27/lstm_cell_27/bias/m
&:$d22Adam/dense_81/kernel/v
 :22Adam/dense_81/bias/v
&:$222Adam/dense_82/kernel/v
 :22Adam/dense_82/bias/v
&:$22Adam/dense_83/kernel/v
 :2Adam/dense_83/bias/v
3:1	2"Adam/lstm_27/lstm_cell_27/kernel/v
=:;	d2,Adam/lstm_27/lstm_cell_27/recurrent_kernel/v
-:+2 Adam/lstm_27/lstm_cell_27/bias/vЁ
!__inference__wrapped_model_939810|	/01%&-.:Ђ7
0Ђ-
+(
lstm_27_inputџџџџџџџџџ
Њ "3Њ0
.
dense_83"
dense_83џџџџџџџџџЄ
D__inference_dense_81_layer_call_and_return_conditional_losses_941790\/Ђ,
%Ђ"
 
inputsџџџџџџџџџd
Њ "%Ђ"

0џџџџџџџџџ2
 |
)__inference_dense_81_layer_call_fn_941779O/Ђ,
%Ђ"
 
inputsџџџџџџџџџd
Њ "џџџџџџџџџ2Є
D__inference_dense_82_layer_call_and_return_conditional_losses_941810\%&/Ђ,
%Ђ"
 
inputsџџџџџџџџџ2
Њ "%Ђ"

0џџџџџџџџџ2
 |
)__inference_dense_82_layer_call_fn_941799O%&/Ђ,
%Ђ"
 
inputsџџџџџџџџџ2
Њ "џџџџџџџџџ2Є
D__inference_dense_83_layer_call_and_return_conditional_losses_941829\-./Ђ,
%Ђ"
 
inputsџџџџџџџџџ2
Њ "%Ђ"

0џџџџџџџџџ
 |
)__inference_dense_83_layer_call_fn_941819O-./Ђ,
%Ђ"
 
inputsџџџџџџџџџ2
Њ "џџџџџџџџџФ
C__inference_lstm_27_layer_call_and_return_conditional_losses_941335}/01OЂL
EЂB
41
/,
inputs/0џџџџџџџџџџџџџџџџџџ

 
p 

 
Њ "%Ђ"

0џџџџџџџџџd
 Ф
C__inference_lstm_27_layer_call_and_return_conditional_losses_941480}/01OЂL
EЂB
41
/,
inputs/0џџџџџџџџџџџџџџџџџџ

 
p

 
Њ "%Ђ"

0џџџџџџџџџd
 Д
C__inference_lstm_27_layer_call_and_return_conditional_losses_941625m/01?Ђ<
5Ђ2
$!
inputsџџџџџџџџџ

 
p 

 
Њ "%Ђ"

0џџџџџџџџџd
 Д
C__inference_lstm_27_layer_call_and_return_conditional_losses_941770m/01?Ђ<
5Ђ2
$!
inputsџџџџџџџџџ

 
p

 
Њ "%Ђ"

0џџџџџџџџџd
 
(__inference_lstm_27_layer_call_fn_941157p/01OЂL
EЂB
41
/,
inputs/0џџџџџџџџџџџџџџџџџџ

 
p 

 
Њ "џџџџџџџџџd
(__inference_lstm_27_layer_call_fn_941168p/01OЂL
EЂB
41
/,
inputs/0џџџџџџџџџџџџџџџџџџ

 
p

 
Њ "џџџџџџџџџd
(__inference_lstm_27_layer_call_fn_941179`/01?Ђ<
5Ђ2
$!
inputsџџџџџџџџџ

 
p 

 
Њ "џџџџџџџџџd
(__inference_lstm_27_layer_call_fn_941190`/01?Ђ<
5Ђ2
$!
inputsџџџџџџџџџ

 
p

 
Њ "џџџџџџџџџdЪ
H__inference_lstm_cell_27_layer_call_and_return_conditional_losses_941895§/01Ђ}
vЂs
 
inputsџџџџџџџџџ
KЂH
"
states/0џџџџџџџџџd
"
states/1џџџџџџџџџd
p 
Њ "sЂp
iЂf

0/0џџџџџџџџџd
EB

0/1/0џџџџџџџџџd

0/1/1џџџџџџџџџd
 Ъ
H__inference_lstm_cell_27_layer_call_and_return_conditional_losses_941927§/01Ђ}
vЂs
 
inputsџџџџџџџџџ
KЂH
"
states/0џџџџџџџџџd
"
states/1џџџџџџџџџd
p
Њ "sЂp
iЂf

0/0џџџџџџџџџd
EB

0/1/0џџџџџџџџџd

0/1/1џџџџџџџџџd
 
-__inference_lstm_cell_27_layer_call_fn_941846э/01Ђ}
vЂs
 
inputsџџџџџџџџџ
KЂH
"
states/0џџџџџџџџџd
"
states/1џџџџџџџџџd
p 
Њ "cЂ`

0џџџџџџџџџd
A>

1/0џџџџџџџџџd

1/1џџџџџџџџџd
-__inference_lstm_cell_27_layer_call_fn_941863э/01Ђ}
vЂs
 
inputsџџџџџџџџџ
KЂH
"
states/0џџџџџџџџџd
"
states/1џџџџџџџџџd
p
Њ "cЂ`

0џџџџџџџџџd
A>

1/0џџџџџџџџџd

1/1џџџџџџџџџdУ
I__inference_sequential_27_layer_call_and_return_conditional_losses_940713v	/01%&-.BЂ?
8Ђ5
+(
lstm_27_inputџџџџџџџџџ
p 

 
Њ "%Ђ"

0џџџџџџџџџ
 У
I__inference_sequential_27_layer_call_and_return_conditional_losses_940739v	/01%&-.BЂ?
8Ђ5
+(
lstm_27_inputџџџџџџџџџ
p

 
Њ "%Ђ"

0џџџџџџџџџ
 М
I__inference_sequential_27_layer_call_and_return_conditional_losses_940981o	/01%&-.;Ђ8
1Ђ.
$!
inputsџџџџџџџџџ
p 

 
Њ "%Ђ"

0џџџџџџџџџ
 М
I__inference_sequential_27_layer_call_and_return_conditional_losses_941146o	/01%&-.;Ђ8
1Ђ.
$!
inputsџџџџџџџџџ
p

 
Њ "%Ђ"

0џџџџџџџџџ
 
.__inference_sequential_27_layer_call_fn_940395i	/01%&-.BЂ?
8Ђ5
+(
lstm_27_inputџџџџџџџџџ
p 

 
Њ "џџџџџџџџџ
.__inference_sequential_27_layer_call_fn_940687i	/01%&-.BЂ?
8Ђ5
+(
lstm_27_inputџџџџџџџџџ
p

 
Њ "џџџџџџџџџ
.__inference_sequential_27_layer_call_fn_940793b	/01%&-.;Ђ8
1Ђ.
$!
inputsџџџџџџџџџ
p 

 
Њ "џџџџџџџџџ
.__inference_sequential_27_layer_call_fn_940816b	/01%&-.;Ђ8
1Ђ.
$!
inputsџџџџџџџџџ
p

 
Њ "џџџџџџџџџЖ
$__inference_signature_wrapper_940770	/01%&-.KЂH
Ђ 
AЊ>
<
lstm_27_input+(
lstm_27_inputџџџџџџџџџ"3Њ0
.
dense_83"
dense_83џџџџџџџџџ