№
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
"serve*2.10.02v2.10.0-rc3-6-g359c3cdfc5f8ђ

!Adam/lstm_5/lstm_cell_2959/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/lstm_5/lstm_cell_2959/bias/v

5Adam/lstm_5/lstm_cell_2959/bias/v/Read/ReadVariableOpReadVariableOp!Adam/lstm_5/lstm_cell_2959/bias/v*
_output_shapes	
:*
dtype0
З
-Adam/lstm_5/lstm_cell_2959/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	d*>
shared_name/-Adam/lstm_5/lstm_cell_2959/recurrent_kernel/v
А
AAdam/lstm_5/lstm_cell_2959/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp-Adam/lstm_5/lstm_cell_2959/recurrent_kernel/v*
_output_shapes
:	d*
dtype0
Ѓ
#Adam/lstm_5/lstm_cell_2959/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*4
shared_name%#Adam/lstm_5/lstm_cell_2959/kernel/v

7Adam/lstm_5/lstm_cell_2959/kernel/v/Read/ReadVariableOpReadVariableOp#Adam/lstm_5/lstm_cell_2959/kernel/v*
_output_shapes
:	*
dtype0

Adam/dense_17/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_17/bias/v
y
(Adam/dense_17/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_17/bias/v*
_output_shapes
:*
dtype0

Adam/dense_17/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*'
shared_nameAdam/dense_17/kernel/v

*Adam/dense_17/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_17/kernel/v*
_output_shapes

:2*
dtype0

Adam/dense_16/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*%
shared_nameAdam/dense_16/bias/v
y
(Adam/dense_16/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_16/bias/v*
_output_shapes
:2*
dtype0

Adam/dense_16/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:22*'
shared_nameAdam/dense_16/kernel/v

*Adam/dense_16/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_16/kernel/v*
_output_shapes

:22*
dtype0

Adam/dense_15/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*%
shared_nameAdam/dense_15/bias/v
y
(Adam/dense_15/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_15/bias/v*
_output_shapes
:2*
dtype0

Adam/dense_15/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d2*'
shared_nameAdam/dense_15/kernel/v

*Adam/dense_15/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_15/kernel/v*
_output_shapes

:d2*
dtype0

!Adam/lstm_5/lstm_cell_2959/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/lstm_5/lstm_cell_2959/bias/m

5Adam/lstm_5/lstm_cell_2959/bias/m/Read/ReadVariableOpReadVariableOp!Adam/lstm_5/lstm_cell_2959/bias/m*
_output_shapes	
:*
dtype0
З
-Adam/lstm_5/lstm_cell_2959/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	d*>
shared_name/-Adam/lstm_5/lstm_cell_2959/recurrent_kernel/m
А
AAdam/lstm_5/lstm_cell_2959/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp-Adam/lstm_5/lstm_cell_2959/recurrent_kernel/m*
_output_shapes
:	d*
dtype0
Ѓ
#Adam/lstm_5/lstm_cell_2959/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*4
shared_name%#Adam/lstm_5/lstm_cell_2959/kernel/m

7Adam/lstm_5/lstm_cell_2959/kernel/m/Read/ReadVariableOpReadVariableOp#Adam/lstm_5/lstm_cell_2959/kernel/m*
_output_shapes
:	*
dtype0

Adam/dense_17/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_17/bias/m
y
(Adam/dense_17/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_17/bias/m*
_output_shapes
:*
dtype0

Adam/dense_17/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*'
shared_nameAdam/dense_17/kernel/m

*Adam/dense_17/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_17/kernel/m*
_output_shapes

:2*
dtype0

Adam/dense_16/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*%
shared_nameAdam/dense_16/bias/m
y
(Adam/dense_16/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_16/bias/m*
_output_shapes
:2*
dtype0

Adam/dense_16/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:22*'
shared_nameAdam/dense_16/kernel/m

*Adam/dense_16/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_16/kernel/m*
_output_shapes

:22*
dtype0

Adam/dense_15/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*%
shared_nameAdam/dense_15/bias/m
y
(Adam/dense_15/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_15/bias/m*
_output_shapes
:2*
dtype0

Adam/dense_15/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d2*'
shared_nameAdam/dense_15/kernel/m

*Adam/dense_15/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_15/kernel/m*
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

lstm_5/lstm_cell_2959/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namelstm_5/lstm_cell_2959/bias

.lstm_5/lstm_cell_2959/bias/Read/ReadVariableOpReadVariableOplstm_5/lstm_cell_2959/bias*
_output_shapes	
:*
dtype0
Љ
&lstm_5/lstm_cell_2959/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	d*7
shared_name(&lstm_5/lstm_cell_2959/recurrent_kernel
Ђ
:lstm_5/lstm_cell_2959/recurrent_kernel/Read/ReadVariableOpReadVariableOp&lstm_5/lstm_cell_2959/recurrent_kernel*
_output_shapes
:	d*
dtype0

lstm_5/lstm_cell_2959/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*-
shared_namelstm_5/lstm_cell_2959/kernel

0lstm_5/lstm_cell_2959/kernel/Read/ReadVariableOpReadVariableOplstm_5/lstm_cell_2959/kernel*
_output_shapes
:	*
dtype0
r
dense_17/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_17/bias
k
!dense_17/bias/Read/ReadVariableOpReadVariableOpdense_17/bias*
_output_shapes
:*
dtype0
z
dense_17/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2* 
shared_namedense_17/kernel
s
#dense_17/kernel/Read/ReadVariableOpReadVariableOpdense_17/kernel*
_output_shapes

:2*
dtype0
r
dense_16/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*
shared_namedense_16/bias
k
!dense_16/bias/Read/ReadVariableOpReadVariableOpdense_16/bias*
_output_shapes
:2*
dtype0
z
dense_16/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:22* 
shared_namedense_16/kernel
s
#dense_16/kernel/Read/ReadVariableOpReadVariableOpdense_16/kernel*
_output_shapes

:22*
dtype0
r
dense_15/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*
shared_namedense_15/bias
k
!dense_15/bias/Read/ReadVariableOpReadVariableOpdense_15/bias*
_output_shapes
:2*
dtype0
z
dense_15/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d2* 
shared_namedense_15/kernel
s
#dense_15/kernel/Read/ReadVariableOpReadVariableOpdense_15/kernel*
_output_shapes

:d2*
dtype0

serving_default_lstm_5_inputPlaceholder*+
_output_shapes
:џџџџџџџџџ*
dtype0* 
shape:џџџџџџџџџ

StatefulPartitionedCallStatefulPartitionedCallserving_default_lstm_5_inputlstm_5/lstm_cell_2959/kernel&lstm_5/lstm_cell_2959/recurrent_kernellstm_5/lstm_cell_2959/biasdense_15/kerneldense_15/biasdense_16/kerneldense_16/biasdense_17/kerneldense_17/bias*
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
GPU 2J 8 */
f*R(
&__inference_signature_wrapper_24307605

NoOpNoOp
ЇF
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*тE
valueиEBеE BЮE

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
signatures
#_self_saveable_object_factories*
ц
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_random_generator
cell

state_spec
#_self_saveable_object_factories*
Ы
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
 bias
#!_self_saveable_object_factories*
Ы
"	variables
#trainable_variables
$regularization_losses
%	keras_api
&__call__
*'&call_and_return_all_conditional_losses

(kernel
)bias
#*_self_saveable_object_factories*
Ы
+	variables
,trainable_variables
-regularization_losses
.	keras_api
/__call__
*0&call_and_return_all_conditional_losses

1kernel
2bias
#3_self_saveable_object_factories*
C
40
51
62
3
 4
(5
)6
17
28*
C
40
51
62
3
 4
(5
)6
17
28*
* 
А
7non_trainable_variables

8layers
9metrics
:layer_regularization_losses
;layer_metrics
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
<trace_0
=trace_1
>trace_2
?trace_3* 
6
@trace_0
Atrace_1
Btrace_2
Ctrace_3* 
* 
ј
Diter

Ebeta_1

Fbeta_2
	Gdecay
Hlearning_ratem m(m)m1m2m4m5m6mv v(v)v1v2v4v5v6v*

Iserving_default* 
* 

40
51
62*

40
51
62*
* 


Jstates
Knon_trainable_variables

Llayers
Mmetrics
Nlayer_regularization_losses
Olayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
Ptrace_0
Qtrace_1
Rtrace_2
Strace_3* 
6
Ttrace_0
Utrace_1
Vtrace_2
Wtrace_3* 
'
#X_self_saveable_object_factories* 

Y	variables
Ztrainable_variables
[regularization_losses
\	keras_api
]__call__
*^&call_and_return_all_conditional_losses
__random_generator
`
state_size

4kernel
5recurrent_kernel
6bias
#a_self_saveable_object_factories*
* 
* 

0
 1*

0
 1*
* 

bnon_trainable_variables

clayers
dmetrics
elayer_regularization_losses
flayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

gtrace_0* 

htrace_0* 
_Y
VARIABLE_VALUEdense_15/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_15/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

(0
)1*

(0
)1*
* 

inon_trainable_variables

jlayers
kmetrics
llayer_regularization_losses
mlayer_metrics
"	variables
#trainable_variables
$regularization_losses
&__call__
*'&call_and_return_all_conditional_losses
&'"call_and_return_conditional_losses*

ntrace_0* 

otrace_0* 
_Y
VARIABLE_VALUEdense_16/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_16/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

10
21*

10
21*
* 

pnon_trainable_variables

qlayers
rmetrics
slayer_regularization_losses
tlayer_metrics
+	variables
,trainable_variables
-regularization_losses
/__call__
*0&call_and_return_all_conditional_losses
&0"call_and_return_conditional_losses*

utrace_0* 

vtrace_0* 
_Y
VARIABLE_VALUEdense_17/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_17/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
\V
VARIABLE_VALUElstm_5/lstm_cell_2959/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUE&lstm_5/lstm_cell_2959/recurrent_kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElstm_5/lstm_cell_2959/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
0
1
2
3*

w0
x1*
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

0*
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

40
51
62*

40
51
62*
* 

ynon_trainable_variables

zlayers
{metrics
|layer_regularization_losses
}layer_metrics
Y	variables
Ztrainable_variables
[regularization_losses
]__call__
*^&call_and_return_all_conditional_losses
&^"call_and_return_conditional_losses*

~trace_0
trace_1* 

trace_0
trace_1* 
(
$_self_saveable_object_factories* 
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
<
	variables
	keras_api

total

count*
M
	variables
	keras_api

total

count

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
* 

0
1*

	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
|
VARIABLE_VALUEAdam/dense_15/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_15/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_16/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_16/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_17/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_17/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE#Adam/lstm_5/lstm_cell_2959/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE-Adam/lstm_5/lstm_cell_2959/recurrent_kernel/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE!Adam/lstm_5/lstm_cell_2959/bias/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_15/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_15/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_16/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_16/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_17/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_17/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE#Adam/lstm_5/lstm_cell_2959/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE-Adam/lstm_5/lstm_cell_2959/recurrent_kernel/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE!Adam/lstm_5/lstm_cell_2959/bias/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
К
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_15/kernel/Read/ReadVariableOp!dense_15/bias/Read/ReadVariableOp#dense_16/kernel/Read/ReadVariableOp!dense_16/bias/Read/ReadVariableOp#dense_17/kernel/Read/ReadVariableOp!dense_17/bias/Read/ReadVariableOp0lstm_5/lstm_cell_2959/kernel/Read/ReadVariableOp:lstm_5/lstm_cell_2959/recurrent_kernel/Read/ReadVariableOp.lstm_5/lstm_cell_2959/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp*Adam/dense_15/kernel/m/Read/ReadVariableOp(Adam/dense_15/bias/m/Read/ReadVariableOp*Adam/dense_16/kernel/m/Read/ReadVariableOp(Adam/dense_16/bias/m/Read/ReadVariableOp*Adam/dense_17/kernel/m/Read/ReadVariableOp(Adam/dense_17/bias/m/Read/ReadVariableOp7Adam/lstm_5/lstm_cell_2959/kernel/m/Read/ReadVariableOpAAdam/lstm_5/lstm_cell_2959/recurrent_kernel/m/Read/ReadVariableOp5Adam/lstm_5/lstm_cell_2959/bias/m/Read/ReadVariableOp*Adam/dense_15/kernel/v/Read/ReadVariableOp(Adam/dense_15/bias/v/Read/ReadVariableOp*Adam/dense_16/kernel/v/Read/ReadVariableOp(Adam/dense_16/bias/v/Read/ReadVariableOp*Adam/dense_17/kernel/v/Read/ReadVariableOp(Adam/dense_17/bias/v/Read/ReadVariableOp7Adam/lstm_5/lstm_cell_2959/kernel/v/Read/ReadVariableOpAAdam/lstm_5/lstm_cell_2959/recurrent_kernel/v/Read/ReadVariableOp5Adam/lstm_5/lstm_cell_2959/bias/v/Read/ReadVariableOpConst*1
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
GPU 2J 8 **
f%R#
!__inference__traced_save_24308893
х
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_15/kerneldense_15/biasdense_16/kerneldense_16/biasdense_17/kerneldense_17/biaslstm_5/lstm_cell_2959/kernel&lstm_5/lstm_cell_2959/recurrent_kernellstm_5/lstm_cell_2959/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotal_1count_1totalcountAdam/dense_15/kernel/mAdam/dense_15/bias/mAdam/dense_16/kernel/mAdam/dense_16/bias/mAdam/dense_17/kernel/mAdam/dense_17/bias/m#Adam/lstm_5/lstm_cell_2959/kernel/m-Adam/lstm_5/lstm_cell_2959/recurrent_kernel/m!Adam/lstm_5/lstm_cell_2959/bias/mAdam/dense_15/kernel/vAdam/dense_15/bias/vAdam/dense_16/kernel/vAdam/dense_16/bias/vAdam/dense_17/kernel/vAdam/dense_17/bias/v#Adam/lstm_5/lstm_cell_2959/kernel/v-Adam/lstm_5/lstm_cell_2959/recurrent_kernel/v!Adam/lstm_5/lstm_cell_2959/bias/v*0
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
GPU 2J 8 *-
f(R&
$__inference__traced_restore_24309011сс
П
Э
while_cond_24307332
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_24307332___redundant_placeholder06
2while_while_cond_24307332___redundant_placeholder16
2while_while_cond_24307332___redundant_placeholder26
2while_while_cond_24307332___redundant_placeholder3
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
№
н
'sequential_5_lstm_5_while_cond_24306539D
@sequential_5_lstm_5_while_sequential_5_lstm_5_while_loop_counterJ
Fsequential_5_lstm_5_while_sequential_5_lstm_5_while_maximum_iterations)
%sequential_5_lstm_5_while_placeholder+
'sequential_5_lstm_5_while_placeholder_1+
'sequential_5_lstm_5_while_placeholder_2+
'sequential_5_lstm_5_while_placeholder_3F
Bsequential_5_lstm_5_while_less_sequential_5_lstm_5_strided_slice_1^
Zsequential_5_lstm_5_while_sequential_5_lstm_5_while_cond_24306539___redundant_placeholder0^
Zsequential_5_lstm_5_while_sequential_5_lstm_5_while_cond_24306539___redundant_placeholder1^
Zsequential_5_lstm_5_while_sequential_5_lstm_5_while_cond_24306539___redundant_placeholder2^
Zsequential_5_lstm_5_while_sequential_5_lstm_5_while_cond_24306539___redundant_placeholder3&
"sequential_5_lstm_5_while_identity
В
sequential_5/lstm_5/while/LessLess%sequential_5_lstm_5_while_placeholderBsequential_5_lstm_5_while_less_sequential_5_lstm_5_strided_slice_1*
T0*
_output_shapes
: s
"sequential_5/lstm_5/while/IdentityIdentity"sequential_5/lstm_5/while/Less:z:0*
T0
*
_output_shapes
: "Q
"sequential_5_lstm_5_while_identity+sequential_5/lstm_5/while/Identity:output:0*(
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
ЅL
Ј
D__inference_lstm_5_layer_call_and_return_conditional_losses_24307150

inputs@
-lstm_cell_3533_matmul_readvariableop_resource:	B
/lstm_cell_3533_matmul_1_readvariableop_resource:	d=
.lstm_cell_3533_biasadd_readvariableop_resource:	
identityЂ%lstm_cell_3533/BiasAdd/ReadVariableOpЂ$lstm_cell_3533/MatMul/ReadVariableOpЂ&lstm_cell_3533/MatMul_1/ReadVariableOpЂwhile;
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
shrink_axis_mask
$lstm_cell_3533/MatMul/ReadVariableOpReadVariableOp-lstm_cell_3533_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
lstm_cell_3533/MatMulMatMulstrided_slice_2:output:0,lstm_cell_3533/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ
&lstm_cell_3533/MatMul_1/ReadVariableOpReadVariableOp/lstm_cell_3533_matmul_1_readvariableop_resource*
_output_shapes
:	d*
dtype0
lstm_cell_3533/MatMul_1MatMulzeros:output:0.lstm_cell_3533/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ
lstm_cell_3533/addAddV2lstm_cell_3533/MatMul:product:0!lstm_cell_3533/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџ
%lstm_cell_3533/BiasAdd/ReadVariableOpReadVariableOp.lstm_cell_3533_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
lstm_cell_3533/BiasAddBiasAddlstm_cell_3533/add:z:0-lstm_cell_3533/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ`
lstm_cell_3533/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :у
lstm_cell_3533/splitSplit'lstm_cell_3533/split/split_dim:output:0lstm_cell_3533/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџd:џџџџџџџџџd:џџџџџџџџџd:џџџџџџџџџd*
	num_splitr
lstm_cell_3533/SigmoidSigmoidlstm_cell_3533/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџdt
lstm_cell_3533/Sigmoid_1Sigmoidlstm_cell_3533/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџd{
lstm_cell_3533/mulMullstm_cell_3533/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџdl
lstm_cell_3533/ReluRelulstm_cell_3533/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџd
lstm_cell_3533/mul_1Mullstm_cell_3533/Sigmoid:y:0!lstm_cell_3533/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџd
lstm_cell_3533/add_1AddV2lstm_cell_3533/mul:z:0lstm_cell_3533/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџdt
lstm_cell_3533/Sigmoid_2Sigmoidlstm_cell_3533/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџdi
lstm_cell_3533/Relu_1Relulstm_cell_3533/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџd
lstm_cell_3533/mul_2Mullstm_cell_3533/Sigmoid_2:y:0#lstm_cell_3533/Relu_1:activations:0*
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
value	B : 
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0-lstm_cell_3533_matmul_readvariableop_resource/lstm_cell_3533_matmul_1_readvariableop_resource.lstm_cell_3533_biasadd_readvariableop_resource*
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
bodyR
while_body_24307065*
condR
while_cond_24307064*K
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
:џџџџџџџџџdЦ
NoOpNoOp&^lstm_cell_3533/BiasAdd/ReadVariableOp%^lstm_cell_3533/MatMul/ReadVariableOp'^lstm_cell_3533/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:џџџџџџџџџ: : : 2N
%lstm_cell_3533/BiasAdd/ReadVariableOp%lstm_cell_3533/BiasAdd/ReadVariableOp2L
$lstm_cell_3533/MatMul/ReadVariableOp$lstm_cell_3533/MatMul/ReadVariableOp2P
&lstm_cell_3533/MatMul_1/ReadVariableOp&lstm_cell_3533/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ѓ
њ
1__inference_lstm_cell_3533_layer_call_fn_24308698

inputs
states_0
states_1
unknown:	
	unknown_0:	d
	unknown_1:	
identity

identity_1

identity_2ЂStatefulPartitionedCallЌ
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
GPU 2J 8 *U
fPRN
L__inference_lstm_cell_3533_layer_call_and_return_conditional_losses_24306860o
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
П
Э
while_cond_24308374
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_24308374___redundant_placeholder06
2while_while_cond_24308374___redundant_placeholder16
2while_while_cond_24308374___redundant_placeholder26
2while_while_cond_24308374___redundant_placeholder3
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


й
lstm_5_while_cond_24307710*
&lstm_5_while_lstm_5_while_loop_counter0
,lstm_5_while_lstm_5_while_maximum_iterations
lstm_5_while_placeholder
lstm_5_while_placeholder_1
lstm_5_while_placeholder_2
lstm_5_while_placeholder_3,
(lstm_5_while_less_lstm_5_strided_slice_1D
@lstm_5_while_lstm_5_while_cond_24307710___redundant_placeholder0D
@lstm_5_while_lstm_5_while_cond_24307710___redundant_placeholder1D
@lstm_5_while_lstm_5_while_cond_24307710___redundant_placeholder2D
@lstm_5_while_lstm_5_while_cond_24307710___redundant_placeholder3
lstm_5_while_identity
~
lstm_5/while/LessLesslstm_5_while_placeholder(lstm_5_while_less_lstm_5_strided_slice_1*
T0*
_output_shapes
: Y
lstm_5/while/IdentityIdentitylstm_5/while/Less:z:0*
T0
*
_output_shapes
: "7
lstm_5_while_identitylstm_5/while/Identity:output:0*(
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
Ц

+__inference_dense_15_layer_call_fn_24308614

inputs
unknown:d2
	unknown_0:2
identityЂStatefulPartitionedCallл
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
GPU 2J 8 *O
fJRH
F__inference_dense_15_layer_call_and_return_conditional_losses_24307169o
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
с

L__inference_lstm_cell_3533_layer_call_and_return_conditional_losses_24308730

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
Д:
ф
while_body_24307333
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0H
5while_lstm_cell_3533_matmul_readvariableop_resource_0:	J
7while_lstm_cell_3533_matmul_1_readvariableop_resource_0:	dE
6while_lstm_cell_3533_biasadd_readvariableop_resource_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorF
3while_lstm_cell_3533_matmul_readvariableop_resource:	H
5while_lstm_cell_3533_matmul_1_readvariableop_resource:	dC
4while_lstm_cell_3533_biasadd_readvariableop_resource:	Ђ+while/lstm_cell_3533/BiasAdd/ReadVariableOpЂ*while/lstm_cell_3533/MatMul/ReadVariableOpЂ,while/lstm_cell_3533/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   І
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype0Ё
*while/lstm_cell_3533/MatMul/ReadVariableOpReadVariableOp5while_lstm_cell_3533_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype0О
while/lstm_cell_3533/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:02while/lstm_cell_3533/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЅ
,while/lstm_cell_3533/MatMul_1/ReadVariableOpReadVariableOp7while_lstm_cell_3533_matmul_1_readvariableop_resource_0*
_output_shapes
:	d*
dtype0Ѕ
while/lstm_cell_3533/MatMul_1MatMulwhile_placeholder_24while/lstm_cell_3533/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЄ
while/lstm_cell_3533/addAddV2%while/lstm_cell_3533/MatMul:product:0'while/lstm_cell_3533/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџ
+while/lstm_cell_3533/BiasAdd/ReadVariableOpReadVariableOp6while_lstm_cell_3533_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0­
while/lstm_cell_3533/BiasAddBiasAddwhile/lstm_cell_3533/add:z:03while/lstm_cell_3533/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџf
$while/lstm_cell_3533/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ѕ
while/lstm_cell_3533/splitSplit-while/lstm_cell_3533/split/split_dim:output:0%while/lstm_cell_3533/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџd:џџџџџџџџџd:џџџџџџџџџd:џџџџџџџџџd*
	num_split~
while/lstm_cell_3533/SigmoidSigmoid#while/lstm_cell_3533/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџd
while/lstm_cell_3533/Sigmoid_1Sigmoid#while/lstm_cell_3533/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџd
while/lstm_cell_3533/mulMul"while/lstm_cell_3533/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџdx
while/lstm_cell_3533/ReluRelu#while/lstm_cell_3533/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџd
while/lstm_cell_3533/mul_1Mul while/lstm_cell_3533/Sigmoid:y:0'while/lstm_cell_3533/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџd
while/lstm_cell_3533/add_1AddV2while/lstm_cell_3533/mul:z:0while/lstm_cell_3533/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџd
while/lstm_cell_3533/Sigmoid_2Sigmoid#while/lstm_cell_3533/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџdu
while/lstm_cell_3533/Relu_1Reluwhile/lstm_cell_3533/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџdЂ
while/lstm_cell_3533/mul_2Mul"while/lstm_cell_3533/Sigmoid_2:y:0)while/lstm_cell_3533/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџdr
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : я
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell_3533/mul_2:z:0*
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
: {
while/Identity_4Identitywhile/lstm_cell_3533/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџd{
while/Identity_5Identitywhile/lstm_cell_3533/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџdж

while/NoOpNoOp,^while/lstm_cell_3533/BiasAdd/ReadVariableOp+^while/lstm_cell_3533/MatMul/ReadVariableOp-^while/lstm_cell_3533/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"n
4while_lstm_cell_3533_biasadd_readvariableop_resource6while_lstm_cell_3533_biasadd_readvariableop_resource_0"p
5while_lstm_cell_3533_matmul_1_readvariableop_resource7while_lstm_cell_3533_matmul_1_readvariableop_resource_0"l
3while_lstm_cell_3533_matmul_readvariableop_resource5while_lstm_cell_3533_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџd:џџџџџџџџџd: : : : : 2Z
+while/lstm_cell_3533/BiasAdd/ReadVariableOp+while/lstm_cell_3533/BiasAdd/ReadVariableOp2X
*while/lstm_cell_3533/MatMul/ReadVariableOp*while/lstm_cell_3533/MatMul/ReadVariableOp2\
,while/lstm_cell_3533/MatMul_1/ReadVariableOp,while/lstm_cell_3533/MatMul_1/ReadVariableOp: 
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
ШL
Њ
D__inference_lstm_5_layer_call_and_return_conditional_losses_24308315
inputs_0@
-lstm_cell_3533_matmul_readvariableop_resource:	B
/lstm_cell_3533_matmul_1_readvariableop_resource:	d=
.lstm_cell_3533_biasadd_readvariableop_resource:	
identityЂ%lstm_cell_3533/BiasAdd/ReadVariableOpЂ$lstm_cell_3533/MatMul/ReadVariableOpЂ&lstm_cell_3533/MatMul_1/ReadVariableOpЂwhile=
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
shrink_axis_mask
$lstm_cell_3533/MatMul/ReadVariableOpReadVariableOp-lstm_cell_3533_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
lstm_cell_3533/MatMulMatMulstrided_slice_2:output:0,lstm_cell_3533/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ
&lstm_cell_3533/MatMul_1/ReadVariableOpReadVariableOp/lstm_cell_3533_matmul_1_readvariableop_resource*
_output_shapes
:	d*
dtype0
lstm_cell_3533/MatMul_1MatMulzeros:output:0.lstm_cell_3533/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ
lstm_cell_3533/addAddV2lstm_cell_3533/MatMul:product:0!lstm_cell_3533/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџ
%lstm_cell_3533/BiasAdd/ReadVariableOpReadVariableOp.lstm_cell_3533_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
lstm_cell_3533/BiasAddBiasAddlstm_cell_3533/add:z:0-lstm_cell_3533/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ`
lstm_cell_3533/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :у
lstm_cell_3533/splitSplit'lstm_cell_3533/split/split_dim:output:0lstm_cell_3533/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџd:џџџџџџџџџd:џџџџџџџџџd:џџџџџџџџџd*
	num_splitr
lstm_cell_3533/SigmoidSigmoidlstm_cell_3533/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџdt
lstm_cell_3533/Sigmoid_1Sigmoidlstm_cell_3533/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџd{
lstm_cell_3533/mulMullstm_cell_3533/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџdl
lstm_cell_3533/ReluRelulstm_cell_3533/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџd
lstm_cell_3533/mul_1Mullstm_cell_3533/Sigmoid:y:0!lstm_cell_3533/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџd
lstm_cell_3533/add_1AddV2lstm_cell_3533/mul:z:0lstm_cell_3533/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџdt
lstm_cell_3533/Sigmoid_2Sigmoidlstm_cell_3533/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџdi
lstm_cell_3533/Relu_1Relulstm_cell_3533/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџd
lstm_cell_3533/mul_2Mullstm_cell_3533/Sigmoid_2:y:0#lstm_cell_3533/Relu_1:activations:0*
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
value	B : 
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0-lstm_cell_3533_matmul_readvariableop_resource/lstm_cell_3533_matmul_1_readvariableop_resource.lstm_cell_3533_biasadd_readvariableop_resource*
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
bodyR
while_body_24308230*
condR
while_cond_24308229*K
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
:џџџџџџџџџdЦ
NoOpNoOp&^lstm_cell_3533/BiasAdd/ReadVariableOp%^lstm_cell_3533/MatMul/ReadVariableOp'^lstm_cell_3533/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџџџџџџџџџџ: : : 2N
%lstm_cell_3533/BiasAdd/ReadVariableOp%lstm_cell_3533/BiasAdd/ReadVariableOp2L
$lstm_cell_3533/MatMul/ReadVariableOp$lstm_cell_3533/MatMul/ReadVariableOp2P
&lstm_cell_3533/MatMul_1/ReadVariableOp&lstm_cell_3533/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/0
љ
Ж
)__inference_lstm_5_layer_call_fn_24308025

inputs
unknown:	
	unknown_0:	d
	unknown_1:	
identityЂStatefulPartitionedCallц
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
GPU 2J 8 *M
fHRF
D__inference_lstm_5_layer_call_and_return_conditional_losses_24307418o
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


ї
F__inference_dense_16_layer_call_and_return_conditional_losses_24308645

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
е
Г
J__inference_sequential_5_layer_call_and_return_conditional_losses_24307548
lstm_5_input"
lstm_5_24307525:	"
lstm_5_24307527:	d
lstm_5_24307529:	#
dense_15_24307532:d2
dense_15_24307534:2#
dense_16_24307537:22
dense_16_24307539:2#
dense_17_24307542:2
dense_17_24307544:
identityЂ dense_15/StatefulPartitionedCallЂ dense_16/StatefulPartitionedCallЂ dense_17/StatefulPartitionedCallЂlstm_5/StatefulPartitionedCall
lstm_5/StatefulPartitionedCallStatefulPartitionedCalllstm_5_inputlstm_5_24307525lstm_5_24307527lstm_5_24307529*
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
GPU 2J 8 *M
fHRF
D__inference_lstm_5_layer_call_and_return_conditional_losses_24307150
 dense_15/StatefulPartitionedCallStatefulPartitionedCall'lstm_5/StatefulPartitionedCall:output:0dense_15_24307532dense_15_24307534*
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
GPU 2J 8 *O
fJRH
F__inference_dense_15_layer_call_and_return_conditional_losses_24307169
 dense_16/StatefulPartitionedCallStatefulPartitionedCall)dense_15/StatefulPartitionedCall:output:0dense_16_24307537dense_16_24307539*
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
GPU 2J 8 *O
fJRH
F__inference_dense_16_layer_call_and_return_conditional_losses_24307186
 dense_17/StatefulPartitionedCallStatefulPartitionedCall)dense_16/StatefulPartitionedCall:output:0dense_17_24307542dense_17_24307544*
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
GPU 2J 8 *O
fJRH
F__inference_dense_17_layer_call_and_return_conditional_losses_24307202x
IdentityIdentity)dense_17/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџа
NoOpNoOp!^dense_15/StatefulPartitionedCall!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall^lstm_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):џџџџџџџџџ: : : : : : : : : 2D
 dense_15/StatefulPartitionedCall dense_15/StatefulPartitionedCall2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall2D
 dense_17/StatefulPartitionedCall dense_17/StatefulPartitionedCall2@
lstm_5/StatefulPartitionedCalllstm_5/StatefulPartitionedCall:Y U
+
_output_shapes
:џџџџџџџџџ
&
_user_specified_namelstm_5_input
ЅL
Ј
D__inference_lstm_5_layer_call_and_return_conditional_losses_24308605

inputs@
-lstm_cell_3533_matmul_readvariableop_resource:	B
/lstm_cell_3533_matmul_1_readvariableop_resource:	d=
.lstm_cell_3533_biasadd_readvariableop_resource:	
identityЂ%lstm_cell_3533/BiasAdd/ReadVariableOpЂ$lstm_cell_3533/MatMul/ReadVariableOpЂ&lstm_cell_3533/MatMul_1/ReadVariableOpЂwhile;
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
shrink_axis_mask
$lstm_cell_3533/MatMul/ReadVariableOpReadVariableOp-lstm_cell_3533_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
lstm_cell_3533/MatMulMatMulstrided_slice_2:output:0,lstm_cell_3533/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ
&lstm_cell_3533/MatMul_1/ReadVariableOpReadVariableOp/lstm_cell_3533_matmul_1_readvariableop_resource*
_output_shapes
:	d*
dtype0
lstm_cell_3533/MatMul_1MatMulzeros:output:0.lstm_cell_3533/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ
lstm_cell_3533/addAddV2lstm_cell_3533/MatMul:product:0!lstm_cell_3533/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџ
%lstm_cell_3533/BiasAdd/ReadVariableOpReadVariableOp.lstm_cell_3533_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
lstm_cell_3533/BiasAddBiasAddlstm_cell_3533/add:z:0-lstm_cell_3533/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ`
lstm_cell_3533/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :у
lstm_cell_3533/splitSplit'lstm_cell_3533/split/split_dim:output:0lstm_cell_3533/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџd:џџџџџџџџџd:џџџџџџџџџd:џџџџџџџџџd*
	num_splitr
lstm_cell_3533/SigmoidSigmoidlstm_cell_3533/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџdt
lstm_cell_3533/Sigmoid_1Sigmoidlstm_cell_3533/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџd{
lstm_cell_3533/mulMullstm_cell_3533/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџdl
lstm_cell_3533/ReluRelulstm_cell_3533/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџd
lstm_cell_3533/mul_1Mullstm_cell_3533/Sigmoid:y:0!lstm_cell_3533/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџd
lstm_cell_3533/add_1AddV2lstm_cell_3533/mul:z:0lstm_cell_3533/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџdt
lstm_cell_3533/Sigmoid_2Sigmoidlstm_cell_3533/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџdi
lstm_cell_3533/Relu_1Relulstm_cell_3533/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџd
lstm_cell_3533/mul_2Mullstm_cell_3533/Sigmoid_2:y:0#lstm_cell_3533/Relu_1:activations:0*
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
value	B : 
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0-lstm_cell_3533_matmul_readvariableop_resource/lstm_cell_3533_matmul_1_readvariableop_resource.lstm_cell_3533_biasadd_readvariableop_resource*
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
bodyR
while_body_24308520*
condR
while_cond_24308519*K
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
:џџџџџџџџџdЦ
NoOpNoOp&^lstm_cell_3533/BiasAdd/ReadVariableOp%^lstm_cell_3533/MatMul/ReadVariableOp'^lstm_cell_3533/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:џџџџџџџџџ: : : 2N
%lstm_cell_3533/BiasAdd/ReadVariableOp%lstm_cell_3533/BiasAdd/ReadVariableOp2L
$lstm_cell_3533/MatMul/ReadVariableOp$lstm_cell_3533/MatMul/ReadVariableOp2P
&lstm_cell_3533/MatMul_1/ReadVariableOp&lstm_cell_3533/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
П
Э
while_cond_24308084
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_24308084___redundant_placeholder06
2while_while_cond_24308084___redundant_placeholder16
2while_while_cond_24308084___redundant_placeholder26
2while_while_cond_24308084___redundant_placeholder3
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
лL
ѓ
!__inference__traced_save_24308893
file_prefix.
*savev2_dense_15_kernel_read_readvariableop,
(savev2_dense_15_bias_read_readvariableop.
*savev2_dense_16_kernel_read_readvariableop,
(savev2_dense_16_bias_read_readvariableop.
*savev2_dense_17_kernel_read_readvariableop,
(savev2_dense_17_bias_read_readvariableop;
7savev2_lstm_5_lstm_cell_2959_kernel_read_readvariableopE
Asavev2_lstm_5_lstm_cell_2959_recurrent_kernel_read_readvariableop9
5savev2_lstm_5_lstm_cell_2959_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop5
1savev2_adam_dense_15_kernel_m_read_readvariableop3
/savev2_adam_dense_15_bias_m_read_readvariableop5
1savev2_adam_dense_16_kernel_m_read_readvariableop3
/savev2_adam_dense_16_bias_m_read_readvariableop5
1savev2_adam_dense_17_kernel_m_read_readvariableop3
/savev2_adam_dense_17_bias_m_read_readvariableopB
>savev2_adam_lstm_5_lstm_cell_2959_kernel_m_read_readvariableopL
Hsavev2_adam_lstm_5_lstm_cell_2959_recurrent_kernel_m_read_readvariableop@
<savev2_adam_lstm_5_lstm_cell_2959_bias_m_read_readvariableop5
1savev2_adam_dense_15_kernel_v_read_readvariableop3
/savev2_adam_dense_15_bias_v_read_readvariableop5
1savev2_adam_dense_16_kernel_v_read_readvariableop3
/savev2_adam_dense_16_bias_v_read_readvariableop5
1savev2_adam_dense_17_kernel_v_read_readvariableop3
/savev2_adam_dense_17_bias_v_read_readvariableopB
>savev2_adam_lstm_5_lstm_cell_2959_kernel_v_read_readvariableopL
Hsavev2_adam_lstm_5_lstm_cell_2959_recurrent_kernel_v_read_readvariableop@
<savev2_adam_lstm_5_lstm_cell_2959_bias_v_read_readvariableop
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
valueTBR%B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B Щ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_15_kernel_read_readvariableop(savev2_dense_15_bias_read_readvariableop*savev2_dense_16_kernel_read_readvariableop(savev2_dense_16_bias_read_readvariableop*savev2_dense_17_kernel_read_readvariableop(savev2_dense_17_bias_read_readvariableop7savev2_lstm_5_lstm_cell_2959_kernel_read_readvariableopAsavev2_lstm_5_lstm_cell_2959_recurrent_kernel_read_readvariableop5savev2_lstm_5_lstm_cell_2959_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop1savev2_adam_dense_15_kernel_m_read_readvariableop/savev2_adam_dense_15_bias_m_read_readvariableop1savev2_adam_dense_16_kernel_m_read_readvariableop/savev2_adam_dense_16_bias_m_read_readvariableop1savev2_adam_dense_17_kernel_m_read_readvariableop/savev2_adam_dense_17_bias_m_read_readvariableop>savev2_adam_lstm_5_lstm_cell_2959_kernel_m_read_readvariableopHsavev2_adam_lstm_5_lstm_cell_2959_recurrent_kernel_m_read_readvariableop<savev2_adam_lstm_5_lstm_cell_2959_bias_m_read_readvariableop1savev2_adam_dense_15_kernel_v_read_readvariableop/savev2_adam_dense_15_bias_v_read_readvariableop1savev2_adam_dense_16_kernel_v_read_readvariableop/savev2_adam_dense_16_bias_v_read_readvariableop1savev2_adam_dense_17_kernel_v_read_readvariableop/savev2_adam_dense_17_bias_v_read_readvariableop>savev2_adam_lstm_5_lstm_cell_2959_kernel_v_read_readvariableopHsavev2_adam_lstm_5_lstm_cell_2959_recurrent_kernel_v_read_readvariableop<savev2_adam_lstm_5_lstm_cell_2959_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
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
ЅL
Ј
D__inference_lstm_5_layer_call_and_return_conditional_losses_24307418

inputs@
-lstm_cell_3533_matmul_readvariableop_resource:	B
/lstm_cell_3533_matmul_1_readvariableop_resource:	d=
.lstm_cell_3533_biasadd_readvariableop_resource:	
identityЂ%lstm_cell_3533/BiasAdd/ReadVariableOpЂ$lstm_cell_3533/MatMul/ReadVariableOpЂ&lstm_cell_3533/MatMul_1/ReadVariableOpЂwhile;
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
shrink_axis_mask
$lstm_cell_3533/MatMul/ReadVariableOpReadVariableOp-lstm_cell_3533_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
lstm_cell_3533/MatMulMatMulstrided_slice_2:output:0,lstm_cell_3533/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ
&lstm_cell_3533/MatMul_1/ReadVariableOpReadVariableOp/lstm_cell_3533_matmul_1_readvariableop_resource*
_output_shapes
:	d*
dtype0
lstm_cell_3533/MatMul_1MatMulzeros:output:0.lstm_cell_3533/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ
lstm_cell_3533/addAddV2lstm_cell_3533/MatMul:product:0!lstm_cell_3533/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџ
%lstm_cell_3533/BiasAdd/ReadVariableOpReadVariableOp.lstm_cell_3533_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
lstm_cell_3533/BiasAddBiasAddlstm_cell_3533/add:z:0-lstm_cell_3533/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ`
lstm_cell_3533/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :у
lstm_cell_3533/splitSplit'lstm_cell_3533/split/split_dim:output:0lstm_cell_3533/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџd:џџџџџџџџџd:џџџџџџџџџd:џџџџџџџџџd*
	num_splitr
lstm_cell_3533/SigmoidSigmoidlstm_cell_3533/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџdt
lstm_cell_3533/Sigmoid_1Sigmoidlstm_cell_3533/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџd{
lstm_cell_3533/mulMullstm_cell_3533/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџdl
lstm_cell_3533/ReluRelulstm_cell_3533/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџd
lstm_cell_3533/mul_1Mullstm_cell_3533/Sigmoid:y:0!lstm_cell_3533/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџd
lstm_cell_3533/add_1AddV2lstm_cell_3533/mul:z:0lstm_cell_3533/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџdt
lstm_cell_3533/Sigmoid_2Sigmoidlstm_cell_3533/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџdi
lstm_cell_3533/Relu_1Relulstm_cell_3533/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџd
lstm_cell_3533/mul_2Mullstm_cell_3533/Sigmoid_2:y:0#lstm_cell_3533/Relu_1:activations:0*
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
value	B : 
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0-lstm_cell_3533_matmul_readvariableop_resource/lstm_cell_3533_matmul_1_readvariableop_resource.lstm_cell_3533_biasadd_readvariableop_resource*
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
bodyR
while_body_24307333*
condR
while_cond_24307332*K
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
:џџџџџџџџџdЦ
NoOpNoOp&^lstm_cell_3533/BiasAdd/ReadVariableOp%^lstm_cell_3533/MatMul/ReadVariableOp'^lstm_cell_3533/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:џџџџџџџџџ: : : 2N
%lstm_cell_3533/BiasAdd/ReadVariableOp%lstm_cell_3533/BiasAdd/ReadVariableOp2L
$lstm_cell_3533/MatMul/ReadVariableOp$lstm_cell_3533/MatMul/ReadVariableOp2P
&lstm_cell_3533/MatMul_1/ReadVariableOp&lstm_cell_3533/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs


й
lstm_5_while_cond_24307875*
&lstm_5_while_lstm_5_while_loop_counter0
,lstm_5_while_lstm_5_while_maximum_iterations
lstm_5_while_placeholder
lstm_5_while_placeholder_1
lstm_5_while_placeholder_2
lstm_5_while_placeholder_3,
(lstm_5_while_less_lstm_5_strided_slice_1D
@lstm_5_while_lstm_5_while_cond_24307875___redundant_placeholder0D
@lstm_5_while_lstm_5_while_cond_24307875___redundant_placeholder1D
@lstm_5_while_lstm_5_while_cond_24307875___redundant_placeholder2D
@lstm_5_while_lstm_5_while_cond_24307875___redundant_placeholder3
lstm_5_while_identity
~
lstm_5/while/LessLesslstm_5_while_placeholder(lstm_5_while_less_lstm_5_strided_slice_1*
T0*
_output_shapes
: Y
lstm_5/while/IdentityIdentitylstm_5/while/Less:z:0*
T0
*
_output_shapes
: "7
lstm_5_while_identitylstm_5/while/Identity:output:0*(
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


ф
/__inference_sequential_5_layer_call_fn_24307230
lstm_5_input
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
StatefulPartitionedCallStatefulPartitionedCalllstm_5_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
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
GPU 2J 8 *S
fNRL
J__inference_sequential_5_layer_call_and_return_conditional_losses_24307209o
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
StatefulPartitionedCallStatefulPartitionedCall:Y U
+
_output_shapes
:џџџџџџџџџ
&
_user_specified_namelstm_5_input

И
)__inference_lstm_5_layer_call_fn_24307992
inputs_0
unknown:	
	unknown_0:	d
	unknown_1:	
identityЂStatefulPartitionedCallш
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
GPU 2J 8 *M
fHRF
D__inference_lstm_5_layer_call_and_return_conditional_losses_24306797o
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

н
$__inference__traced_restore_24309011
file_prefix2
 assignvariableop_dense_15_kernel:d2.
 assignvariableop_1_dense_15_bias:24
"assignvariableop_2_dense_16_kernel:22.
 assignvariableop_3_dense_16_bias:24
"assignvariableop_4_dense_17_kernel:2.
 assignvariableop_5_dense_17_bias:B
/assignvariableop_6_lstm_5_lstm_cell_2959_kernel:	L
9assignvariableop_7_lstm_5_lstm_cell_2959_recurrent_kernel:	d<
-assignvariableop_8_lstm_5_lstm_cell_2959_bias:	&
assignvariableop_9_adam_iter:	 )
assignvariableop_10_adam_beta_1: )
assignvariableop_11_adam_beta_2: (
assignvariableop_12_adam_decay: 0
&assignvariableop_13_adam_learning_rate: %
assignvariableop_14_total_1: %
assignvariableop_15_count_1: #
assignvariableop_16_total: #
assignvariableop_17_count: <
*assignvariableop_18_adam_dense_15_kernel_m:d26
(assignvariableop_19_adam_dense_15_bias_m:2<
*assignvariableop_20_adam_dense_16_kernel_m:226
(assignvariableop_21_adam_dense_16_bias_m:2<
*assignvariableop_22_adam_dense_17_kernel_m:26
(assignvariableop_23_adam_dense_17_bias_m:J
7assignvariableop_24_adam_lstm_5_lstm_cell_2959_kernel_m:	T
Aassignvariableop_25_adam_lstm_5_lstm_cell_2959_recurrent_kernel_m:	dD
5assignvariableop_26_adam_lstm_5_lstm_cell_2959_bias_m:	<
*assignvariableop_27_adam_dense_15_kernel_v:d26
(assignvariableop_28_adam_dense_15_bias_v:2<
*assignvariableop_29_adam_dense_16_kernel_v:226
(assignvariableop_30_adam_dense_16_bias_v:2<
*assignvariableop_31_adam_dense_17_kernel_v:26
(assignvariableop_32_adam_dense_17_bias_v:J
7assignvariableop_33_adam_lstm_5_lstm_cell_2959_kernel_v:	T
Aassignvariableop_34_adam_lstm_5_lstm_cell_2959_recurrent_kernel_v:	dD
5assignvariableop_35_adam_lstm_5_lstm_cell_2959_bias_v:	
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
AssignVariableOpAssignVariableOp assignvariableop_dense_15_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_15_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOp"assignvariableop_2_dense_16_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOp assignvariableop_3_dense_16_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOp"assignvariableop_4_dense_17_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOp assignvariableop_5_dense_17_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOp/assignvariableop_6_lstm_5_lstm_cell_2959_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:Ј
AssignVariableOp_7AssignVariableOp9assignvariableop_7_lstm_5_lstm_cell_2959_recurrent_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_8AssignVariableOp-assignvariableop_8_lstm_5_lstm_cell_2959_biasIdentity_8:output:0"/device:CPU:0*
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
AssignVariableOp_18AssignVariableOp*assignvariableop_18_adam_dense_15_kernel_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_19AssignVariableOp(assignvariableop_19_adam_dense_15_bias_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_20AssignVariableOp*assignvariableop_20_adam_dense_16_kernel_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_21AssignVariableOp(assignvariableop_21_adam_dense_16_bias_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_22AssignVariableOp*assignvariableop_22_adam_dense_17_kernel_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_23AssignVariableOp(assignvariableop_23_adam_dense_17_bias_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:Ј
AssignVariableOp_24AssignVariableOp7assignvariableop_24_adam_lstm_5_lstm_cell_2959_kernel_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:В
AssignVariableOp_25AssignVariableOpAassignvariableop_25_adam_lstm_5_lstm_cell_2959_recurrent_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:І
AssignVariableOp_26AssignVariableOp5assignvariableop_26_adam_lstm_5_lstm_cell_2959_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_27AssignVariableOp*assignvariableop_27_adam_dense_15_kernel_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_28AssignVariableOp(assignvariableop_28_adam_dense_15_bias_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_29AssignVariableOp*assignvariableop_29_adam_dense_16_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_30AssignVariableOp(assignvariableop_30_adam_dense_16_bias_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_31AssignVariableOp*assignvariableop_31_adam_dense_17_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_32AssignVariableOp(assignvariableop_32_adam_dense_17_bias_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:Ј
AssignVariableOp_33AssignVariableOp7assignvariableop_33_adam_lstm_5_lstm_cell_2959_kernel_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:В
AssignVariableOp_34AssignVariableOpAassignvariableop_34_adam_lstm_5_lstm_cell_2959_recurrent_kernel_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:І
AssignVariableOp_35AssignVariableOp5assignvariableop_35_adam_lstm_5_lstm_cell_2959_bias_vIdentity_35:output:0"/device:CPU:0*
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
ђl

J__inference_sequential_5_layer_call_and_return_conditional_losses_24307981

inputsG
4lstm_5_lstm_cell_3533_matmul_readvariableop_resource:	I
6lstm_5_lstm_cell_3533_matmul_1_readvariableop_resource:	dD
5lstm_5_lstm_cell_3533_biasadd_readvariableop_resource:	9
'dense_15_matmul_readvariableop_resource:d26
(dense_15_biasadd_readvariableop_resource:29
'dense_16_matmul_readvariableop_resource:226
(dense_16_biasadd_readvariableop_resource:29
'dense_17_matmul_readvariableop_resource:26
(dense_17_biasadd_readvariableop_resource:
identityЂdense_15/BiasAdd/ReadVariableOpЂdense_15/MatMul/ReadVariableOpЂdense_16/BiasAdd/ReadVariableOpЂdense_16/MatMul/ReadVariableOpЂdense_17/BiasAdd/ReadVariableOpЂdense_17/MatMul/ReadVariableOpЂ,lstm_5/lstm_cell_3533/BiasAdd/ReadVariableOpЂ+lstm_5/lstm_cell_3533/MatMul/ReadVariableOpЂ-lstm_5/lstm_cell_3533/MatMul_1/ReadVariableOpЂlstm_5/whileB
lstm_5/ShapeShapeinputs*
T0*
_output_shapes
:d
lstm_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: f
lstm_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:f
lstm_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:є
lstm_5/strided_sliceStridedSlicelstm_5/Shape:output:0#lstm_5/strided_slice/stack:output:0%lstm_5/strided_slice/stack_1:output:0%lstm_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskW
lstm_5/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d
lstm_5/zeros/packedPacklstm_5/strided_slice:output:0lstm_5/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:W
lstm_5/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
lstm_5/zerosFilllstm_5/zeros/packed:output:0lstm_5/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџdY
lstm_5/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d
lstm_5/zeros_1/packedPacklstm_5/strided_slice:output:0 lstm_5/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:Y
lstm_5/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
lstm_5/zeros_1Filllstm_5/zeros_1/packed:output:0lstm_5/zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџdj
lstm_5/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          {
lstm_5/transpose	Transposeinputslstm_5/transpose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџR
lstm_5/Shape_1Shapelstm_5/transpose:y:0*
T0*
_output_shapes
:f
lstm_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: h
lstm_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
lstm_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ў
lstm_5/strided_slice_1StridedSlicelstm_5/Shape_1:output:0%lstm_5/strided_slice_1/stack:output:0'lstm_5/strided_slice_1/stack_1:output:0'lstm_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskm
"lstm_5/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџЩ
lstm_5/TensorArrayV2TensorListReserve+lstm_5/TensorArrayV2/element_shape:output:0lstm_5/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
<lstm_5/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   ѕ
.lstm_5/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_5/transpose:y:0Elstm_5/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвf
lstm_5/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: h
lstm_5/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
lstm_5/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
lstm_5/strided_slice_2StridedSlicelstm_5/transpose:y:0%lstm_5/strided_slice_2/stack:output:0'lstm_5/strided_slice_2/stack_1:output:0'lstm_5/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_maskЁ
+lstm_5/lstm_cell_3533/MatMul/ReadVariableOpReadVariableOp4lstm_5_lstm_cell_3533_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0Џ
lstm_5/lstm_cell_3533/MatMulMatMullstm_5/strided_slice_2:output:03lstm_5/lstm_cell_3533/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЅ
-lstm_5/lstm_cell_3533/MatMul_1/ReadVariableOpReadVariableOp6lstm_5_lstm_cell_3533_matmul_1_readvariableop_resource*
_output_shapes
:	d*
dtype0Љ
lstm_5/lstm_cell_3533/MatMul_1MatMullstm_5/zeros:output:05lstm_5/lstm_cell_3533/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЇ
lstm_5/lstm_cell_3533/addAddV2&lstm_5/lstm_cell_3533/MatMul:product:0(lstm_5/lstm_cell_3533/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџ
,lstm_5/lstm_cell_3533/BiasAdd/ReadVariableOpReadVariableOp5lstm_5_lstm_cell_3533_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0А
lstm_5/lstm_cell_3533/BiasAddBiasAddlstm_5/lstm_cell_3533/add:z:04lstm_5/lstm_cell_3533/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџg
%lstm_5/lstm_cell_3533/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ј
lstm_5/lstm_cell_3533/splitSplit.lstm_5/lstm_cell_3533/split/split_dim:output:0&lstm_5/lstm_cell_3533/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџd:џџџџџџџџџd:џџџџџџџџџd:џџџџџџџџџd*
	num_split
lstm_5/lstm_cell_3533/SigmoidSigmoid$lstm_5/lstm_cell_3533/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџd
lstm_5/lstm_cell_3533/Sigmoid_1Sigmoid$lstm_5/lstm_cell_3533/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџd
lstm_5/lstm_cell_3533/mulMul#lstm_5/lstm_cell_3533/Sigmoid_1:y:0lstm_5/zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџdz
lstm_5/lstm_cell_3533/ReluRelu$lstm_5/lstm_cell_3533/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџdЁ
lstm_5/lstm_cell_3533/mul_1Mul!lstm_5/lstm_cell_3533/Sigmoid:y:0(lstm_5/lstm_cell_3533/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџd
lstm_5/lstm_cell_3533/add_1AddV2lstm_5/lstm_cell_3533/mul:z:0lstm_5/lstm_cell_3533/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџd
lstm_5/lstm_cell_3533/Sigmoid_2Sigmoid$lstm_5/lstm_cell_3533/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџdw
lstm_5/lstm_cell_3533/Relu_1Relulstm_5/lstm_cell_3533/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџdЅ
lstm_5/lstm_cell_3533/mul_2Mul#lstm_5/lstm_cell_3533/Sigmoid_2:y:0*lstm_5/lstm_cell_3533/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџdu
$lstm_5/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџd   e
#lstm_5/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :к
lstm_5/TensorArrayV2_1TensorListReserve-lstm_5/TensorArrayV2_1/element_shape:output:0,lstm_5/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвM
lstm_5/timeConst*
_output_shapes
: *
dtype0*
value	B : j
lstm_5/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ[
lstm_5/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ю
lstm_5/whileWhile"lstm_5/while/loop_counter:output:0(lstm_5/while/maximum_iterations:output:0lstm_5/time:output:0lstm_5/TensorArrayV2_1:handle:0lstm_5/zeros:output:0lstm_5/zeros_1:output:0lstm_5/strided_slice_1:output:0>lstm_5/TensorArrayUnstack/TensorListFromTensor:output_handle:04lstm_5_lstm_cell_3533_matmul_readvariableop_resource6lstm_5_lstm_cell_3533_matmul_1_readvariableop_resource5lstm_5_lstm_cell_3533_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџd:џџџџџџџџџd: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *&
bodyR
lstm_5_while_body_24307876*&
condR
lstm_5_while_cond_24307875*K
output_shapes:
8: : : : :џџџџџџџџџd:џџџџџџџџџd: : : : : *
parallel_iterations 
7lstm_5/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџd   ы
)lstm_5/TensorArrayV2Stack/TensorListStackTensorListStacklstm_5/while:output:3@lstm_5/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџd*
element_dtype0*
num_elementso
lstm_5/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџh
lstm_5/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: h
lstm_5/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Њ
lstm_5/strided_slice_3StridedSlice2lstm_5/TensorArrayV2Stack/TensorListStack:tensor:0%lstm_5/strided_slice_3/stack:output:0'lstm_5/strided_slice_3/stack_1:output:0'lstm_5/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџd*
shrink_axis_maskl
lstm_5/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ћ
lstm_5/transpose_1	Transpose2lstm_5/TensorArrayV2Stack/TensorListStack:tensor:0 lstm_5/transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџdb
lstm_5/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    
dense_15/MatMul/ReadVariableOpReadVariableOp'dense_15_matmul_readvariableop_resource*
_output_shapes

:d2*
dtype0
dense_15/MatMulMatMullstm_5/strided_slice_3:output:0&dense_15/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_15/BiasAdd/ReadVariableOpReadVariableOp(dense_15_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0
dense_15/BiasAddBiasAdddense_15/MatMul:product:0'dense_15/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2b
dense_15/ReluReludense_15/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_16/MatMul/ReadVariableOpReadVariableOp'dense_16_matmul_readvariableop_resource*
_output_shapes

:22*
dtype0
dense_16/MatMulMatMuldense_15/Relu:activations:0&dense_16/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_16/BiasAdd/ReadVariableOpReadVariableOp(dense_16_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0
dense_16/BiasAddBiasAdddense_16/MatMul:product:0'dense_16/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2b
dense_16/ReluReludense_16/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_17/MatMul/ReadVariableOpReadVariableOp'dense_17_matmul_readvariableop_resource*
_output_shapes

:2*
dtype0
dense_17/MatMulMatMuldense_16/Relu:activations:0&dense_17/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
dense_17/BiasAdd/ReadVariableOpReadVariableOp(dense_17_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_17/BiasAddBiasAdddense_17/MatMul:product:0'dense_17/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџh
IdentityIdentitydense_17/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџЋ
NoOpNoOp ^dense_15/BiasAdd/ReadVariableOp^dense_15/MatMul/ReadVariableOp ^dense_16/BiasAdd/ReadVariableOp^dense_16/MatMul/ReadVariableOp ^dense_17/BiasAdd/ReadVariableOp^dense_17/MatMul/ReadVariableOp-^lstm_5/lstm_cell_3533/BiasAdd/ReadVariableOp,^lstm_5/lstm_cell_3533/MatMul/ReadVariableOp.^lstm_5/lstm_cell_3533/MatMul_1/ReadVariableOp^lstm_5/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):џџџџџџџџџ: : : : : : : : : 2B
dense_15/BiasAdd/ReadVariableOpdense_15/BiasAdd/ReadVariableOp2@
dense_15/MatMul/ReadVariableOpdense_15/MatMul/ReadVariableOp2B
dense_16/BiasAdd/ReadVariableOpdense_16/BiasAdd/ReadVariableOp2@
dense_16/MatMul/ReadVariableOpdense_16/MatMul/ReadVariableOp2B
dense_17/BiasAdd/ReadVariableOpdense_17/BiasAdd/ReadVariableOp2@
dense_17/MatMul/ReadVariableOpdense_17/MatMul/ReadVariableOp2\
,lstm_5/lstm_cell_3533/BiasAdd/ReadVariableOp,lstm_5/lstm_cell_3533/BiasAdd/ReadVariableOp2Z
+lstm_5/lstm_cell_3533/MatMul/ReadVariableOp+lstm_5/lstm_cell_3533/MatMul/ReadVariableOp2^
-lstm_5/lstm_cell_3533/MatMul_1/ReadVariableOp-lstm_5/lstm_cell_3533/MatMul_1/ReadVariableOp2
lstm_5/whilelstm_5/while:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
с

L__inference_lstm_cell_3533_layer_call_and_return_conditional_losses_24308762

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


ф
/__inference_sequential_5_layer_call_fn_24307522
lstm_5_input
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
StatefulPartitionedCallStatefulPartitionedCalllstm_5_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
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
GPU 2J 8 *S
fNRL
J__inference_sequential_5_layer_call_and_return_conditional_losses_24307478o
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
StatefulPartitionedCallStatefulPartitionedCall:Y U
+
_output_shapes
:џџџџџџџџџ
&
_user_specified_namelstm_5_input
Д:
ф
while_body_24308520
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0H
5while_lstm_cell_3533_matmul_readvariableop_resource_0:	J
7while_lstm_cell_3533_matmul_1_readvariableop_resource_0:	dE
6while_lstm_cell_3533_biasadd_readvariableop_resource_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorF
3while_lstm_cell_3533_matmul_readvariableop_resource:	H
5while_lstm_cell_3533_matmul_1_readvariableop_resource:	dC
4while_lstm_cell_3533_biasadd_readvariableop_resource:	Ђ+while/lstm_cell_3533/BiasAdd/ReadVariableOpЂ*while/lstm_cell_3533/MatMul/ReadVariableOpЂ,while/lstm_cell_3533/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   І
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype0Ё
*while/lstm_cell_3533/MatMul/ReadVariableOpReadVariableOp5while_lstm_cell_3533_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype0О
while/lstm_cell_3533/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:02while/lstm_cell_3533/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЅ
,while/lstm_cell_3533/MatMul_1/ReadVariableOpReadVariableOp7while_lstm_cell_3533_matmul_1_readvariableop_resource_0*
_output_shapes
:	d*
dtype0Ѕ
while/lstm_cell_3533/MatMul_1MatMulwhile_placeholder_24while/lstm_cell_3533/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЄ
while/lstm_cell_3533/addAddV2%while/lstm_cell_3533/MatMul:product:0'while/lstm_cell_3533/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџ
+while/lstm_cell_3533/BiasAdd/ReadVariableOpReadVariableOp6while_lstm_cell_3533_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0­
while/lstm_cell_3533/BiasAddBiasAddwhile/lstm_cell_3533/add:z:03while/lstm_cell_3533/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџf
$while/lstm_cell_3533/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ѕ
while/lstm_cell_3533/splitSplit-while/lstm_cell_3533/split/split_dim:output:0%while/lstm_cell_3533/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџd:џџџџџџџџџd:џџџџџџџџџd:џџџџџџџџџd*
	num_split~
while/lstm_cell_3533/SigmoidSigmoid#while/lstm_cell_3533/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџd
while/lstm_cell_3533/Sigmoid_1Sigmoid#while/lstm_cell_3533/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџd
while/lstm_cell_3533/mulMul"while/lstm_cell_3533/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџdx
while/lstm_cell_3533/ReluRelu#while/lstm_cell_3533/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџd
while/lstm_cell_3533/mul_1Mul while/lstm_cell_3533/Sigmoid:y:0'while/lstm_cell_3533/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџd
while/lstm_cell_3533/add_1AddV2while/lstm_cell_3533/mul:z:0while/lstm_cell_3533/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџd
while/lstm_cell_3533/Sigmoid_2Sigmoid#while/lstm_cell_3533/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџdu
while/lstm_cell_3533/Relu_1Reluwhile/lstm_cell_3533/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџdЂ
while/lstm_cell_3533/mul_2Mul"while/lstm_cell_3533/Sigmoid_2:y:0)while/lstm_cell_3533/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџdr
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : я
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell_3533/mul_2:z:0*
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
: {
while/Identity_4Identitywhile/lstm_cell_3533/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџd{
while/Identity_5Identitywhile/lstm_cell_3533/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџdж

while/NoOpNoOp,^while/lstm_cell_3533/BiasAdd/ReadVariableOp+^while/lstm_cell_3533/MatMul/ReadVariableOp-^while/lstm_cell_3533/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"n
4while_lstm_cell_3533_biasadd_readvariableop_resource6while_lstm_cell_3533_biasadd_readvariableop_resource_0"p
5while_lstm_cell_3533_matmul_1_readvariableop_resource7while_lstm_cell_3533_matmul_1_readvariableop_resource_0"l
3while_lstm_cell_3533_matmul_readvariableop_resource5while_lstm_cell_3533_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџd:џџџџџџџџџd: : : : : 2Z
+while/lstm_cell_3533/BiasAdd/ReadVariableOp+while/lstm_cell_3533/BiasAdd/ReadVariableOp2X
*while/lstm_cell_3533/MatMul/ReadVariableOp*while/lstm_cell_3533/MatMul/ReadVariableOp2\
,while/lstm_cell_3533/MatMul_1/ReadVariableOp,while/lstm_cell_3533/MatMul_1/ReadVariableOp: 
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
е
Г
J__inference_sequential_5_layer_call_and_return_conditional_losses_24307574
lstm_5_input"
lstm_5_24307551:	"
lstm_5_24307553:	d
lstm_5_24307555:	#
dense_15_24307558:d2
dense_15_24307560:2#
dense_16_24307563:22
dense_16_24307565:2#
dense_17_24307568:2
dense_17_24307570:
identityЂ dense_15/StatefulPartitionedCallЂ dense_16/StatefulPartitionedCallЂ dense_17/StatefulPartitionedCallЂlstm_5/StatefulPartitionedCall
lstm_5/StatefulPartitionedCallStatefulPartitionedCalllstm_5_inputlstm_5_24307551lstm_5_24307553lstm_5_24307555*
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
GPU 2J 8 *M
fHRF
D__inference_lstm_5_layer_call_and_return_conditional_losses_24307418
 dense_15/StatefulPartitionedCallStatefulPartitionedCall'lstm_5/StatefulPartitionedCall:output:0dense_15_24307558dense_15_24307560*
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
GPU 2J 8 *O
fJRH
F__inference_dense_15_layer_call_and_return_conditional_losses_24307169
 dense_16/StatefulPartitionedCallStatefulPartitionedCall)dense_15/StatefulPartitionedCall:output:0dense_16_24307563dense_16_24307565*
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
GPU 2J 8 *O
fJRH
F__inference_dense_16_layer_call_and_return_conditional_losses_24307186
 dense_17/StatefulPartitionedCallStatefulPartitionedCall)dense_16/StatefulPartitionedCall:output:0dense_17_24307568dense_17_24307570*
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
GPU 2J 8 *O
fJRH
F__inference_dense_17_layer_call_and_return_conditional_losses_24307202x
IdentityIdentity)dense_17/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџа
NoOpNoOp!^dense_15/StatefulPartitionedCall!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall^lstm_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):џџџџџџџџџ: : : : : : : : : 2D
 dense_15/StatefulPartitionedCall dense_15/StatefulPartitionedCall2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall2D
 dense_17/StatefulPartitionedCall dense_17/StatefulPartitionedCall2@
lstm_5/StatefulPartitionedCalllstm_5/StatefulPartitionedCall:Y U
+
_output_shapes
:џџџџџџџџџ
&
_user_specified_namelstm_5_input
У
­
J__inference_sequential_5_layer_call_and_return_conditional_losses_24307478

inputs"
lstm_5_24307455:	"
lstm_5_24307457:	d
lstm_5_24307459:	#
dense_15_24307462:d2
dense_15_24307464:2#
dense_16_24307467:22
dense_16_24307469:2#
dense_17_24307472:2
dense_17_24307474:
identityЂ dense_15/StatefulPartitionedCallЂ dense_16/StatefulPartitionedCallЂ dense_17/StatefulPartitionedCallЂlstm_5/StatefulPartitionedCall
lstm_5/StatefulPartitionedCallStatefulPartitionedCallinputslstm_5_24307455lstm_5_24307457lstm_5_24307459*
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
GPU 2J 8 *M
fHRF
D__inference_lstm_5_layer_call_and_return_conditional_losses_24307418
 dense_15/StatefulPartitionedCallStatefulPartitionedCall'lstm_5/StatefulPartitionedCall:output:0dense_15_24307462dense_15_24307464*
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
GPU 2J 8 *O
fJRH
F__inference_dense_15_layer_call_and_return_conditional_losses_24307169
 dense_16/StatefulPartitionedCallStatefulPartitionedCall)dense_15/StatefulPartitionedCall:output:0dense_16_24307467dense_16_24307469*
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
GPU 2J 8 *O
fJRH
F__inference_dense_16_layer_call_and_return_conditional_losses_24307186
 dense_17/StatefulPartitionedCallStatefulPartitionedCall)dense_16/StatefulPartitionedCall:output:0dense_17_24307472dense_17_24307474*
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
GPU 2J 8 *O
fJRH
F__inference_dense_17_layer_call_and_return_conditional_losses_24307202x
IdentityIdentity)dense_17/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџа
NoOpNoOp!^dense_15/StatefulPartitionedCall!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall^lstm_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):џџџџџџџџџ: : : : : : : : : 2D
 dense_15/StatefulPartitionedCall dense_15/StatefulPartitionedCall2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall2D
 dense_17/StatefulPartitionedCall dense_17/StatefulPartitionedCall2@
lstm_5/StatefulPartitionedCalllstm_5/StatefulPartitionedCall:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ЅL
Ј
D__inference_lstm_5_layer_call_and_return_conditional_losses_24308460

inputs@
-lstm_cell_3533_matmul_readvariableop_resource:	B
/lstm_cell_3533_matmul_1_readvariableop_resource:	d=
.lstm_cell_3533_biasadd_readvariableop_resource:	
identityЂ%lstm_cell_3533/BiasAdd/ReadVariableOpЂ$lstm_cell_3533/MatMul/ReadVariableOpЂ&lstm_cell_3533/MatMul_1/ReadVariableOpЂwhile;
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
shrink_axis_mask
$lstm_cell_3533/MatMul/ReadVariableOpReadVariableOp-lstm_cell_3533_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
lstm_cell_3533/MatMulMatMulstrided_slice_2:output:0,lstm_cell_3533/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ
&lstm_cell_3533/MatMul_1/ReadVariableOpReadVariableOp/lstm_cell_3533_matmul_1_readvariableop_resource*
_output_shapes
:	d*
dtype0
lstm_cell_3533/MatMul_1MatMulzeros:output:0.lstm_cell_3533/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ
lstm_cell_3533/addAddV2lstm_cell_3533/MatMul:product:0!lstm_cell_3533/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџ
%lstm_cell_3533/BiasAdd/ReadVariableOpReadVariableOp.lstm_cell_3533_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
lstm_cell_3533/BiasAddBiasAddlstm_cell_3533/add:z:0-lstm_cell_3533/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ`
lstm_cell_3533/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :у
lstm_cell_3533/splitSplit'lstm_cell_3533/split/split_dim:output:0lstm_cell_3533/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџd:џџџџџџџџџd:џџџџџџџџџd:џџџџџџџџџd*
	num_splitr
lstm_cell_3533/SigmoidSigmoidlstm_cell_3533/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџdt
lstm_cell_3533/Sigmoid_1Sigmoidlstm_cell_3533/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџd{
lstm_cell_3533/mulMullstm_cell_3533/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџdl
lstm_cell_3533/ReluRelulstm_cell_3533/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџd
lstm_cell_3533/mul_1Mullstm_cell_3533/Sigmoid:y:0!lstm_cell_3533/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџd
lstm_cell_3533/add_1AddV2lstm_cell_3533/mul:z:0lstm_cell_3533/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџdt
lstm_cell_3533/Sigmoid_2Sigmoidlstm_cell_3533/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџdi
lstm_cell_3533/Relu_1Relulstm_cell_3533/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџd
lstm_cell_3533/mul_2Mullstm_cell_3533/Sigmoid_2:y:0#lstm_cell_3533/Relu_1:activations:0*
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
value	B : 
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0-lstm_cell_3533_matmul_readvariableop_resource/lstm_cell_3533_matmul_1_readvariableop_resource.lstm_cell_3533_biasadd_readvariableop_resource*
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
bodyR
while_body_24308375*
condR
while_cond_24308374*K
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
:џџџџџџџџџdЦ
NoOpNoOp&^lstm_cell_3533/BiasAdd/ReadVariableOp%^lstm_cell_3533/MatMul/ReadVariableOp'^lstm_cell_3533/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:џџџџџџџџџ: : : 2N
%lstm_cell_3533/BiasAdd/ReadVariableOp%lstm_cell_3533/BiasAdd/ReadVariableOp2L
$lstm_cell_3533/MatMul/ReadVariableOp$lstm_cell_3533/MatMul/ReadVariableOp2P
&lstm_cell_3533/MatMul_1/ReadVariableOp&lstm_cell_3533/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
й

L__inference_lstm_cell_3533_layer_call_and_return_conditional_losses_24306712

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


о
/__inference_sequential_5_layer_call_fn_24307651

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
identityЂStatefulPartitionedCallК
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
GPU 2J 8 *S
fNRL
J__inference_sequential_5_layer_call_and_return_conditional_losses_24307478o
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
У
­
J__inference_sequential_5_layer_call_and_return_conditional_losses_24307209

inputs"
lstm_5_24307151:	"
lstm_5_24307153:	d
lstm_5_24307155:	#
dense_15_24307170:d2
dense_15_24307172:2#
dense_16_24307187:22
dense_16_24307189:2#
dense_17_24307203:2
dense_17_24307205:
identityЂ dense_15/StatefulPartitionedCallЂ dense_16/StatefulPartitionedCallЂ dense_17/StatefulPartitionedCallЂlstm_5/StatefulPartitionedCall
lstm_5/StatefulPartitionedCallStatefulPartitionedCallinputslstm_5_24307151lstm_5_24307153lstm_5_24307155*
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
GPU 2J 8 *M
fHRF
D__inference_lstm_5_layer_call_and_return_conditional_losses_24307150
 dense_15/StatefulPartitionedCallStatefulPartitionedCall'lstm_5/StatefulPartitionedCall:output:0dense_15_24307170dense_15_24307172*
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
GPU 2J 8 *O
fJRH
F__inference_dense_15_layer_call_and_return_conditional_losses_24307169
 dense_16/StatefulPartitionedCallStatefulPartitionedCall)dense_15/StatefulPartitionedCall:output:0dense_16_24307187dense_16_24307189*
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
GPU 2J 8 *O
fJRH
F__inference_dense_16_layer_call_and_return_conditional_losses_24307186
 dense_17/StatefulPartitionedCallStatefulPartitionedCall)dense_16/StatefulPartitionedCall:output:0dense_17_24307203dense_17_24307205*
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
GPU 2J 8 *O
fJRH
F__inference_dense_17_layer_call_and_return_conditional_losses_24307202x
IdentityIdentity)dense_17/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџа
NoOpNoOp!^dense_15/StatefulPartitionedCall!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall^lstm_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):џџџџџџџџџ: : : : : : : : : 2D
 dense_15/StatefulPartitionedCall dense_15/StatefulPartitionedCall2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall2D
 dense_17/StatefulPartitionedCall dense_17/StatefulPartitionedCall2@
lstm_5/StatefulPartitionedCalllstm_5/StatefulPartitionedCall:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Щ	
ї
F__inference_dense_17_layer_call_and_return_conditional_losses_24308664

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
љ
Ж
)__inference_lstm_5_layer_call_fn_24308014

inputs
unknown:	
	unknown_0:	d
	unknown_1:	
identityЂStatefulPartitionedCallц
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
GPU 2J 8 *M
fHRF
D__inference_lstm_5_layer_call_and_return_conditional_losses_24307150o
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
ѓ
њ
1__inference_lstm_cell_3533_layer_call_fn_24308681

inputs
states_0
states_1
unknown:	
	unknown_0:	d
	unknown_1:	
identity

identity_1

identity_2ЂStatefulPartitionedCallЌ
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
GPU 2J 8 *U
fPRN
L__inference_lstm_cell_3533_layer_call_and_return_conditional_losses_24306712o
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
д$
џ
while_body_24306727
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_02
while_lstm_cell_3533_24306751_0:	2
while_lstm_cell_3533_24306753_0:	d.
while_lstm_cell_3533_24306755_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor0
while_lstm_cell_3533_24306751:	0
while_lstm_cell_3533_24306753:	d,
while_lstm_cell_3533_24306755:	Ђ,while/lstm_cell_3533/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   І
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype0Х
,while/lstm_cell_3533/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_3533_24306751_0while_lstm_cell_3533_24306753_0while_lstm_cell_3533_24306755_0*
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
GPU 2J 8 *U
fPRN
L__inference_lstm_cell_3533_layer_call_and_return_conditional_losses_24306712r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : 
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:05while/lstm_cell_3533/StatefulPartitionedCall:output:0*
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
: 
while/Identity_4Identity5while/lstm_cell_3533/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџd
while/Identity_5Identity5while/lstm_cell_3533/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџd{

while/NoOpNoOp-^while/lstm_cell_3533/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"@
while_lstm_cell_3533_24306751while_lstm_cell_3533_24306751_0"@
while_lstm_cell_3533_24306753while_lstm_cell_3533_24306753_0"@
while_lstm_cell_3533_24306755while_lstm_cell_3533_24306755_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџd:џџџџџџџџџd: : : : : 2\
,while/lstm_cell_3533/StatefulPartitionedCall,while/lstm_cell_3533/StatefulPartitionedCall: 
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


ї
F__inference_dense_15_layer_call_and_return_conditional_losses_24307169

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
ШL
Њ
D__inference_lstm_5_layer_call_and_return_conditional_losses_24308170
inputs_0@
-lstm_cell_3533_matmul_readvariableop_resource:	B
/lstm_cell_3533_matmul_1_readvariableop_resource:	d=
.lstm_cell_3533_biasadd_readvariableop_resource:	
identityЂ%lstm_cell_3533/BiasAdd/ReadVariableOpЂ$lstm_cell_3533/MatMul/ReadVariableOpЂ&lstm_cell_3533/MatMul_1/ReadVariableOpЂwhile=
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
shrink_axis_mask
$lstm_cell_3533/MatMul/ReadVariableOpReadVariableOp-lstm_cell_3533_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
lstm_cell_3533/MatMulMatMulstrided_slice_2:output:0,lstm_cell_3533/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ
&lstm_cell_3533/MatMul_1/ReadVariableOpReadVariableOp/lstm_cell_3533_matmul_1_readvariableop_resource*
_output_shapes
:	d*
dtype0
lstm_cell_3533/MatMul_1MatMulzeros:output:0.lstm_cell_3533/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ
lstm_cell_3533/addAddV2lstm_cell_3533/MatMul:product:0!lstm_cell_3533/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџ
%lstm_cell_3533/BiasAdd/ReadVariableOpReadVariableOp.lstm_cell_3533_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
lstm_cell_3533/BiasAddBiasAddlstm_cell_3533/add:z:0-lstm_cell_3533/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ`
lstm_cell_3533/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :у
lstm_cell_3533/splitSplit'lstm_cell_3533/split/split_dim:output:0lstm_cell_3533/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџd:џџџџџџџџџd:џџџџџџџџџd:џџџџџџџџџd*
	num_splitr
lstm_cell_3533/SigmoidSigmoidlstm_cell_3533/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџdt
lstm_cell_3533/Sigmoid_1Sigmoidlstm_cell_3533/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџd{
lstm_cell_3533/mulMullstm_cell_3533/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџdl
lstm_cell_3533/ReluRelulstm_cell_3533/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџd
lstm_cell_3533/mul_1Mullstm_cell_3533/Sigmoid:y:0!lstm_cell_3533/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџd
lstm_cell_3533/add_1AddV2lstm_cell_3533/mul:z:0lstm_cell_3533/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџdt
lstm_cell_3533/Sigmoid_2Sigmoidlstm_cell_3533/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџdi
lstm_cell_3533/Relu_1Relulstm_cell_3533/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџd
lstm_cell_3533/mul_2Mullstm_cell_3533/Sigmoid_2:y:0#lstm_cell_3533/Relu_1:activations:0*
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
value	B : 
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0-lstm_cell_3533_matmul_readvariableop_resource/lstm_cell_3533_matmul_1_readvariableop_resource.lstm_cell_3533_biasadd_readvariableop_resource*
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
bodyR
while_body_24308085*
condR
while_cond_24308084*K
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
:џџџџџџџџџdЦ
NoOpNoOp&^lstm_cell_3533/BiasAdd/ReadVariableOp%^lstm_cell_3533/MatMul/ReadVariableOp'^lstm_cell_3533/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџџџџџџџџџџ: : : 2N
%lstm_cell_3533/BiasAdd/ReadVariableOp%lstm_cell_3533/BiasAdd/ReadVariableOp2L
$lstm_cell_3533/MatMul/ReadVariableOp$lstm_cell_3533/MatMul/ReadVariableOp2P
&lstm_cell_3533/MatMul_1/ReadVariableOp&lstm_cell_3533/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/0
П
Э
while_cond_24307064
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_24307064___redundant_placeholder06
2while_while_cond_24307064___redundant_placeholder16
2while_while_cond_24307064___redundant_placeholder26
2while_while_cond_24307064___redundant_placeholder3
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
Д:
ф
while_body_24308230
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0H
5while_lstm_cell_3533_matmul_readvariableop_resource_0:	J
7while_lstm_cell_3533_matmul_1_readvariableop_resource_0:	dE
6while_lstm_cell_3533_biasadd_readvariableop_resource_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorF
3while_lstm_cell_3533_matmul_readvariableop_resource:	H
5while_lstm_cell_3533_matmul_1_readvariableop_resource:	dC
4while_lstm_cell_3533_biasadd_readvariableop_resource:	Ђ+while/lstm_cell_3533/BiasAdd/ReadVariableOpЂ*while/lstm_cell_3533/MatMul/ReadVariableOpЂ,while/lstm_cell_3533/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   І
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype0Ё
*while/lstm_cell_3533/MatMul/ReadVariableOpReadVariableOp5while_lstm_cell_3533_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype0О
while/lstm_cell_3533/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:02while/lstm_cell_3533/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЅ
,while/lstm_cell_3533/MatMul_1/ReadVariableOpReadVariableOp7while_lstm_cell_3533_matmul_1_readvariableop_resource_0*
_output_shapes
:	d*
dtype0Ѕ
while/lstm_cell_3533/MatMul_1MatMulwhile_placeholder_24while/lstm_cell_3533/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЄ
while/lstm_cell_3533/addAddV2%while/lstm_cell_3533/MatMul:product:0'while/lstm_cell_3533/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџ
+while/lstm_cell_3533/BiasAdd/ReadVariableOpReadVariableOp6while_lstm_cell_3533_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0­
while/lstm_cell_3533/BiasAddBiasAddwhile/lstm_cell_3533/add:z:03while/lstm_cell_3533/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџf
$while/lstm_cell_3533/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ѕ
while/lstm_cell_3533/splitSplit-while/lstm_cell_3533/split/split_dim:output:0%while/lstm_cell_3533/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџd:џџџџџџџџџd:џџџџџџџџџd:џџџџџџџџџd*
	num_split~
while/lstm_cell_3533/SigmoidSigmoid#while/lstm_cell_3533/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџd
while/lstm_cell_3533/Sigmoid_1Sigmoid#while/lstm_cell_3533/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџd
while/lstm_cell_3533/mulMul"while/lstm_cell_3533/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџdx
while/lstm_cell_3533/ReluRelu#while/lstm_cell_3533/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџd
while/lstm_cell_3533/mul_1Mul while/lstm_cell_3533/Sigmoid:y:0'while/lstm_cell_3533/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџd
while/lstm_cell_3533/add_1AddV2while/lstm_cell_3533/mul:z:0while/lstm_cell_3533/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџd
while/lstm_cell_3533/Sigmoid_2Sigmoid#while/lstm_cell_3533/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџdu
while/lstm_cell_3533/Relu_1Reluwhile/lstm_cell_3533/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџdЂ
while/lstm_cell_3533/mul_2Mul"while/lstm_cell_3533/Sigmoid_2:y:0)while/lstm_cell_3533/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџdr
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : я
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell_3533/mul_2:z:0*
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
: {
while/Identity_4Identitywhile/lstm_cell_3533/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџd{
while/Identity_5Identitywhile/lstm_cell_3533/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџdж

while/NoOpNoOp,^while/lstm_cell_3533/BiasAdd/ReadVariableOp+^while/lstm_cell_3533/MatMul/ReadVariableOp-^while/lstm_cell_3533/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"n
4while_lstm_cell_3533_biasadd_readvariableop_resource6while_lstm_cell_3533_biasadd_readvariableop_resource_0"p
5while_lstm_cell_3533_matmul_1_readvariableop_resource7while_lstm_cell_3533_matmul_1_readvariableop_resource_0"l
3while_lstm_cell_3533_matmul_readvariableop_resource5while_lstm_cell_3533_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџd:џџџџџџџџџd: : : : : 2Z
+while/lstm_cell_3533/BiasAdd/ReadVariableOp+while/lstm_cell_3533/BiasAdd/ReadVariableOp2X
*while/lstm_cell_3533/MatMul/ReadVariableOp*while/lstm_cell_3533/MatMul/ReadVariableOp2\
,while/lstm_cell_3533/MatMul_1/ReadVariableOp,while/lstm_cell_3533/MatMul_1/ReadVariableOp: 
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
мB
Ф

lstm_5_while_body_24307711*
&lstm_5_while_lstm_5_while_loop_counter0
,lstm_5_while_lstm_5_while_maximum_iterations
lstm_5_while_placeholder
lstm_5_while_placeholder_1
lstm_5_while_placeholder_2
lstm_5_while_placeholder_3)
%lstm_5_while_lstm_5_strided_slice_1_0e
alstm_5_while_tensorarrayv2read_tensorlistgetitem_lstm_5_tensorarrayunstack_tensorlistfromtensor_0O
<lstm_5_while_lstm_cell_3533_matmul_readvariableop_resource_0:	Q
>lstm_5_while_lstm_cell_3533_matmul_1_readvariableop_resource_0:	dL
=lstm_5_while_lstm_cell_3533_biasadd_readvariableop_resource_0:	
lstm_5_while_identity
lstm_5_while_identity_1
lstm_5_while_identity_2
lstm_5_while_identity_3
lstm_5_while_identity_4
lstm_5_while_identity_5'
#lstm_5_while_lstm_5_strided_slice_1c
_lstm_5_while_tensorarrayv2read_tensorlistgetitem_lstm_5_tensorarrayunstack_tensorlistfromtensorM
:lstm_5_while_lstm_cell_3533_matmul_readvariableop_resource:	O
<lstm_5_while_lstm_cell_3533_matmul_1_readvariableop_resource:	dJ
;lstm_5_while_lstm_cell_3533_biasadd_readvariableop_resource:	Ђ2lstm_5/while/lstm_cell_3533/BiasAdd/ReadVariableOpЂ1lstm_5/while/lstm_cell_3533/MatMul/ReadVariableOpЂ3lstm_5/while/lstm_cell_3533/MatMul_1/ReadVariableOp
>lstm_5/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   Щ
0lstm_5/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemalstm_5_while_tensorarrayv2read_tensorlistgetitem_lstm_5_tensorarrayunstack_tensorlistfromtensor_0lstm_5_while_placeholderGlstm_5/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype0Џ
1lstm_5/while/lstm_cell_3533/MatMul/ReadVariableOpReadVariableOp<lstm_5_while_lstm_cell_3533_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype0г
"lstm_5/while/lstm_cell_3533/MatMulMatMul7lstm_5/while/TensorArrayV2Read/TensorListGetItem:item:09lstm_5/while/lstm_cell_3533/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџГ
3lstm_5/while/lstm_cell_3533/MatMul_1/ReadVariableOpReadVariableOp>lstm_5_while_lstm_cell_3533_matmul_1_readvariableop_resource_0*
_output_shapes
:	d*
dtype0К
$lstm_5/while/lstm_cell_3533/MatMul_1MatMullstm_5_while_placeholder_2;lstm_5/while/lstm_cell_3533/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЙ
lstm_5/while/lstm_cell_3533/addAddV2,lstm_5/while/lstm_cell_3533/MatMul:product:0.lstm_5/while/lstm_cell_3533/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџ­
2lstm_5/while/lstm_cell_3533/BiasAdd/ReadVariableOpReadVariableOp=lstm_5_while_lstm_cell_3533_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0Т
#lstm_5/while/lstm_cell_3533/BiasAddBiasAdd#lstm_5/while/lstm_cell_3533/add:z:0:lstm_5/while/lstm_cell_3533/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџm
+lstm_5/while/lstm_cell_3533/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
!lstm_5/while/lstm_cell_3533/splitSplit4lstm_5/while/lstm_cell_3533/split/split_dim:output:0,lstm_5/while/lstm_cell_3533/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџd:џџџџџџџџџd:џџџџџџџџџd:џџџџџџџџџd*
	num_split
#lstm_5/while/lstm_cell_3533/SigmoidSigmoid*lstm_5/while/lstm_cell_3533/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџd
%lstm_5/while/lstm_cell_3533/Sigmoid_1Sigmoid*lstm_5/while/lstm_cell_3533/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџd
lstm_5/while/lstm_cell_3533/mulMul)lstm_5/while/lstm_cell_3533/Sigmoid_1:y:0lstm_5_while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџd
 lstm_5/while/lstm_cell_3533/ReluRelu*lstm_5/while/lstm_cell_3533/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџdГ
!lstm_5/while/lstm_cell_3533/mul_1Mul'lstm_5/while/lstm_cell_3533/Sigmoid:y:0.lstm_5/while/lstm_cell_3533/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџdЈ
!lstm_5/while/lstm_cell_3533/add_1AddV2#lstm_5/while/lstm_cell_3533/mul:z:0%lstm_5/while/lstm_cell_3533/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџd
%lstm_5/while/lstm_cell_3533/Sigmoid_2Sigmoid*lstm_5/while/lstm_cell_3533/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџd
"lstm_5/while/lstm_cell_3533/Relu_1Relu%lstm_5/while/lstm_cell_3533/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџdЗ
!lstm_5/while/lstm_cell_3533/mul_2Mul)lstm_5/while/lstm_cell_3533/Sigmoid_2:y:00lstm_5/while/lstm_cell_3533/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџdy
7lstm_5/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : 
1lstm_5/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_5_while_placeholder_1@lstm_5/while/TensorArrayV2Write/TensorListSetItem/index:output:0%lstm_5/while/lstm_cell_3533/mul_2:z:0*
_output_shapes
: *
element_dtype0:щшвT
lstm_5/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :q
lstm_5/while/addAddV2lstm_5_while_placeholderlstm_5/while/add/y:output:0*
T0*
_output_shapes
: V
lstm_5/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
lstm_5/while/add_1AddV2&lstm_5_while_lstm_5_while_loop_counterlstm_5/while/add_1/y:output:0*
T0*
_output_shapes
: n
lstm_5/while/IdentityIdentitylstm_5/while/add_1:z:0^lstm_5/while/NoOp*
T0*
_output_shapes
: 
lstm_5/while/Identity_1Identity,lstm_5_while_lstm_5_while_maximum_iterations^lstm_5/while/NoOp*
T0*
_output_shapes
: n
lstm_5/while/Identity_2Identitylstm_5/while/add:z:0^lstm_5/while/NoOp*
T0*
_output_shapes
: 
lstm_5/while/Identity_3IdentityAlstm_5/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_5/while/NoOp*
T0*
_output_shapes
: 
lstm_5/while/Identity_4Identity%lstm_5/while/lstm_cell_3533/mul_2:z:0^lstm_5/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџd
lstm_5/while/Identity_5Identity%lstm_5/while/lstm_cell_3533/add_1:z:0^lstm_5/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџdђ
lstm_5/while/NoOpNoOp3^lstm_5/while/lstm_cell_3533/BiasAdd/ReadVariableOp2^lstm_5/while/lstm_cell_3533/MatMul/ReadVariableOp4^lstm_5/while/lstm_cell_3533/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "7
lstm_5_while_identitylstm_5/while/Identity:output:0";
lstm_5_while_identity_1 lstm_5/while/Identity_1:output:0";
lstm_5_while_identity_2 lstm_5/while/Identity_2:output:0";
lstm_5_while_identity_3 lstm_5/while/Identity_3:output:0";
lstm_5_while_identity_4 lstm_5/while/Identity_4:output:0";
lstm_5_while_identity_5 lstm_5/while/Identity_5:output:0"L
#lstm_5_while_lstm_5_strided_slice_1%lstm_5_while_lstm_5_strided_slice_1_0"|
;lstm_5_while_lstm_cell_3533_biasadd_readvariableop_resource=lstm_5_while_lstm_cell_3533_biasadd_readvariableop_resource_0"~
<lstm_5_while_lstm_cell_3533_matmul_1_readvariableop_resource>lstm_5_while_lstm_cell_3533_matmul_1_readvariableop_resource_0"z
:lstm_5_while_lstm_cell_3533_matmul_readvariableop_resource<lstm_5_while_lstm_cell_3533_matmul_readvariableop_resource_0"Ф
_lstm_5_while_tensorarrayv2read_tensorlistgetitem_lstm_5_tensorarrayunstack_tensorlistfromtensoralstm_5_while_tensorarrayv2read_tensorlistgetitem_lstm_5_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџd:џџџџџџџџџd: : : : : 2h
2lstm_5/while/lstm_cell_3533/BiasAdd/ReadVariableOp2lstm_5/while/lstm_cell_3533/BiasAdd/ReadVariableOp2f
1lstm_5/while/lstm_cell_3533/MatMul/ReadVariableOp1lstm_5/while/lstm_cell_3533/MatMul/ReadVariableOp2j
3lstm_5/while/lstm_cell_3533/MatMul_1/ReadVariableOp3lstm_5/while/lstm_cell_3533/MatMul_1/ReadVariableOp: 
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
R
ф
'sequential_5_lstm_5_while_body_24306540D
@sequential_5_lstm_5_while_sequential_5_lstm_5_while_loop_counterJ
Fsequential_5_lstm_5_while_sequential_5_lstm_5_while_maximum_iterations)
%sequential_5_lstm_5_while_placeholder+
'sequential_5_lstm_5_while_placeholder_1+
'sequential_5_lstm_5_while_placeholder_2+
'sequential_5_lstm_5_while_placeholder_3C
?sequential_5_lstm_5_while_sequential_5_lstm_5_strided_slice_1_0
{sequential_5_lstm_5_while_tensorarrayv2read_tensorlistgetitem_sequential_5_lstm_5_tensorarrayunstack_tensorlistfromtensor_0\
Isequential_5_lstm_5_while_lstm_cell_3533_matmul_readvariableop_resource_0:	^
Ksequential_5_lstm_5_while_lstm_cell_3533_matmul_1_readvariableop_resource_0:	dY
Jsequential_5_lstm_5_while_lstm_cell_3533_biasadd_readvariableop_resource_0:	&
"sequential_5_lstm_5_while_identity(
$sequential_5_lstm_5_while_identity_1(
$sequential_5_lstm_5_while_identity_2(
$sequential_5_lstm_5_while_identity_3(
$sequential_5_lstm_5_while_identity_4(
$sequential_5_lstm_5_while_identity_5A
=sequential_5_lstm_5_while_sequential_5_lstm_5_strided_slice_1}
ysequential_5_lstm_5_while_tensorarrayv2read_tensorlistgetitem_sequential_5_lstm_5_tensorarrayunstack_tensorlistfromtensorZ
Gsequential_5_lstm_5_while_lstm_cell_3533_matmul_readvariableop_resource:	\
Isequential_5_lstm_5_while_lstm_cell_3533_matmul_1_readvariableop_resource:	dW
Hsequential_5_lstm_5_while_lstm_cell_3533_biasadd_readvariableop_resource:	Ђ?sequential_5/lstm_5/while/lstm_cell_3533/BiasAdd/ReadVariableOpЂ>sequential_5/lstm_5/while/lstm_cell_3533/MatMul/ReadVariableOpЂ@sequential_5/lstm_5/while/lstm_cell_3533/MatMul_1/ReadVariableOp
Ksequential_5/lstm_5/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   
=sequential_5/lstm_5/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem{sequential_5_lstm_5_while_tensorarrayv2read_tensorlistgetitem_sequential_5_lstm_5_tensorarrayunstack_tensorlistfromtensor_0%sequential_5_lstm_5_while_placeholderTsequential_5/lstm_5/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype0Щ
>sequential_5/lstm_5/while/lstm_cell_3533/MatMul/ReadVariableOpReadVariableOpIsequential_5_lstm_5_while_lstm_cell_3533_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype0њ
/sequential_5/lstm_5/while/lstm_cell_3533/MatMulMatMulDsequential_5/lstm_5/while/TensorArrayV2Read/TensorListGetItem:item:0Fsequential_5/lstm_5/while/lstm_cell_3533/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЭ
@sequential_5/lstm_5/while/lstm_cell_3533/MatMul_1/ReadVariableOpReadVariableOpKsequential_5_lstm_5_while_lstm_cell_3533_matmul_1_readvariableop_resource_0*
_output_shapes
:	d*
dtype0с
1sequential_5/lstm_5/while/lstm_cell_3533/MatMul_1MatMul'sequential_5_lstm_5_while_placeholder_2Hsequential_5/lstm_5/while/lstm_cell_3533/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџр
,sequential_5/lstm_5/while/lstm_cell_3533/addAddV29sequential_5/lstm_5/while/lstm_cell_3533/MatMul:product:0;sequential_5/lstm_5/while/lstm_cell_3533/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџЧ
?sequential_5/lstm_5/while/lstm_cell_3533/BiasAdd/ReadVariableOpReadVariableOpJsequential_5_lstm_5_while_lstm_cell_3533_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0щ
0sequential_5/lstm_5/while/lstm_cell_3533/BiasAddBiasAdd0sequential_5/lstm_5/while/lstm_cell_3533/add:z:0Gsequential_5/lstm_5/while/lstm_cell_3533/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџz
8sequential_5/lstm_5/while/lstm_cell_3533/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Б
.sequential_5/lstm_5/while/lstm_cell_3533/splitSplitAsequential_5/lstm_5/while/lstm_cell_3533/split/split_dim:output:09sequential_5/lstm_5/while/lstm_cell_3533/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџd:џџџџџџџџџd:џџџџџџџџџd:џџџџџџџџџd*
	num_splitІ
0sequential_5/lstm_5/while/lstm_cell_3533/SigmoidSigmoid7sequential_5/lstm_5/while/lstm_cell_3533/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџdЈ
2sequential_5/lstm_5/while/lstm_cell_3533/Sigmoid_1Sigmoid7sequential_5/lstm_5/while/lstm_cell_3533/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџdЦ
,sequential_5/lstm_5/while/lstm_cell_3533/mulMul6sequential_5/lstm_5/while/lstm_cell_3533/Sigmoid_1:y:0'sequential_5_lstm_5_while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџd 
-sequential_5/lstm_5/while/lstm_cell_3533/ReluRelu7sequential_5/lstm_5/while/lstm_cell_3533/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџdк
.sequential_5/lstm_5/while/lstm_cell_3533/mul_1Mul4sequential_5/lstm_5/while/lstm_cell_3533/Sigmoid:y:0;sequential_5/lstm_5/while/lstm_cell_3533/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџdЯ
.sequential_5/lstm_5/while/lstm_cell_3533/add_1AddV20sequential_5/lstm_5/while/lstm_cell_3533/mul:z:02sequential_5/lstm_5/while/lstm_cell_3533/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџdЈ
2sequential_5/lstm_5/while/lstm_cell_3533/Sigmoid_2Sigmoid7sequential_5/lstm_5/while/lstm_cell_3533/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџd
/sequential_5/lstm_5/while/lstm_cell_3533/Relu_1Relu2sequential_5/lstm_5/while/lstm_cell_3533/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџdо
.sequential_5/lstm_5/while/lstm_cell_3533/mul_2Mul6sequential_5/lstm_5/while/lstm_cell_3533/Sigmoid_2:y:0=sequential_5/lstm_5/while/lstm_cell_3533/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџd
Dsequential_5/lstm_5/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : П
>sequential_5/lstm_5/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem'sequential_5_lstm_5_while_placeholder_1Msequential_5/lstm_5/while/TensorArrayV2Write/TensorListSetItem/index:output:02sequential_5/lstm_5/while/lstm_cell_3533/mul_2:z:0*
_output_shapes
: *
element_dtype0:щшвa
sequential_5/lstm_5/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
sequential_5/lstm_5/while/addAddV2%sequential_5_lstm_5_while_placeholder(sequential_5/lstm_5/while/add/y:output:0*
T0*
_output_shapes
: c
!sequential_5/lstm_5/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :З
sequential_5/lstm_5/while/add_1AddV2@sequential_5_lstm_5_while_sequential_5_lstm_5_while_loop_counter*sequential_5/lstm_5/while/add_1/y:output:0*
T0*
_output_shapes
: 
"sequential_5/lstm_5/while/IdentityIdentity#sequential_5/lstm_5/while/add_1:z:0^sequential_5/lstm_5/while/NoOp*
T0*
_output_shapes
: К
$sequential_5/lstm_5/while/Identity_1IdentityFsequential_5_lstm_5_while_sequential_5_lstm_5_while_maximum_iterations^sequential_5/lstm_5/while/NoOp*
T0*
_output_shapes
: 
$sequential_5/lstm_5/while/Identity_2Identity!sequential_5/lstm_5/while/add:z:0^sequential_5/lstm_5/while/NoOp*
T0*
_output_shapes
: Т
$sequential_5/lstm_5/while/Identity_3IdentityNsequential_5/lstm_5/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^sequential_5/lstm_5/while/NoOp*
T0*
_output_shapes
: З
$sequential_5/lstm_5/while/Identity_4Identity2sequential_5/lstm_5/while/lstm_cell_3533/mul_2:z:0^sequential_5/lstm_5/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџdЗ
$sequential_5/lstm_5/while/Identity_5Identity2sequential_5/lstm_5/while/lstm_cell_3533/add_1:z:0^sequential_5/lstm_5/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџdІ
sequential_5/lstm_5/while/NoOpNoOp@^sequential_5/lstm_5/while/lstm_cell_3533/BiasAdd/ReadVariableOp?^sequential_5/lstm_5/while/lstm_cell_3533/MatMul/ReadVariableOpA^sequential_5/lstm_5/while/lstm_cell_3533/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "Q
"sequential_5_lstm_5_while_identity+sequential_5/lstm_5/while/Identity:output:0"U
$sequential_5_lstm_5_while_identity_1-sequential_5/lstm_5/while/Identity_1:output:0"U
$sequential_5_lstm_5_while_identity_2-sequential_5/lstm_5/while/Identity_2:output:0"U
$sequential_5_lstm_5_while_identity_3-sequential_5/lstm_5/while/Identity_3:output:0"U
$sequential_5_lstm_5_while_identity_4-sequential_5/lstm_5/while/Identity_4:output:0"U
$sequential_5_lstm_5_while_identity_5-sequential_5/lstm_5/while/Identity_5:output:0"
Hsequential_5_lstm_5_while_lstm_cell_3533_biasadd_readvariableop_resourceJsequential_5_lstm_5_while_lstm_cell_3533_biasadd_readvariableop_resource_0"
Isequential_5_lstm_5_while_lstm_cell_3533_matmul_1_readvariableop_resourceKsequential_5_lstm_5_while_lstm_cell_3533_matmul_1_readvariableop_resource_0"
Gsequential_5_lstm_5_while_lstm_cell_3533_matmul_readvariableop_resourceIsequential_5_lstm_5_while_lstm_cell_3533_matmul_readvariableop_resource_0"
=sequential_5_lstm_5_while_sequential_5_lstm_5_strided_slice_1?sequential_5_lstm_5_while_sequential_5_lstm_5_strided_slice_1_0"ј
ysequential_5_lstm_5_while_tensorarrayv2read_tensorlistgetitem_sequential_5_lstm_5_tensorarrayunstack_tensorlistfromtensor{sequential_5_lstm_5_while_tensorarrayv2read_tensorlistgetitem_sequential_5_lstm_5_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџd:џџџџџџџџџd: : : : : 2
?sequential_5/lstm_5/while/lstm_cell_3533/BiasAdd/ReadVariableOp?sequential_5/lstm_5/while/lstm_cell_3533/BiasAdd/ReadVariableOp2
>sequential_5/lstm_5/while/lstm_cell_3533/MatMul/ReadVariableOp>sequential_5/lstm_5/while/lstm_cell_3533/MatMul/ReadVariableOp2
@sequential_5/lstm_5/while/lstm_cell_3533/MatMul_1/ReadVariableOp@sequential_5/lstm_5/while/lstm_cell_3533/MatMul_1/ReadVariableOp: 
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
П
Э
while_cond_24308229
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_24308229___redundant_placeholder06
2while_while_cond_24308229___redundant_placeholder16
2while_while_cond_24308229___redundant_placeholder26
2while_while_cond_24308229___redundant_placeholder3
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
Ц

+__inference_dense_17_layer_call_fn_24308654

inputs
unknown:2
	unknown_0:
identityЂStatefulPartitionedCallл
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
GPU 2J 8 *O
fJRH
F__inference_dense_17_layer_call_and_return_conditional_losses_24307202o
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
Д:
ф
while_body_24308085
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0H
5while_lstm_cell_3533_matmul_readvariableop_resource_0:	J
7while_lstm_cell_3533_matmul_1_readvariableop_resource_0:	dE
6while_lstm_cell_3533_biasadd_readvariableop_resource_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorF
3while_lstm_cell_3533_matmul_readvariableop_resource:	H
5while_lstm_cell_3533_matmul_1_readvariableop_resource:	dC
4while_lstm_cell_3533_biasadd_readvariableop_resource:	Ђ+while/lstm_cell_3533/BiasAdd/ReadVariableOpЂ*while/lstm_cell_3533/MatMul/ReadVariableOpЂ,while/lstm_cell_3533/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   І
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype0Ё
*while/lstm_cell_3533/MatMul/ReadVariableOpReadVariableOp5while_lstm_cell_3533_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype0О
while/lstm_cell_3533/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:02while/lstm_cell_3533/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЅ
,while/lstm_cell_3533/MatMul_1/ReadVariableOpReadVariableOp7while_lstm_cell_3533_matmul_1_readvariableop_resource_0*
_output_shapes
:	d*
dtype0Ѕ
while/lstm_cell_3533/MatMul_1MatMulwhile_placeholder_24while/lstm_cell_3533/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЄ
while/lstm_cell_3533/addAddV2%while/lstm_cell_3533/MatMul:product:0'while/lstm_cell_3533/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџ
+while/lstm_cell_3533/BiasAdd/ReadVariableOpReadVariableOp6while_lstm_cell_3533_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0­
while/lstm_cell_3533/BiasAddBiasAddwhile/lstm_cell_3533/add:z:03while/lstm_cell_3533/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџf
$while/lstm_cell_3533/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ѕ
while/lstm_cell_3533/splitSplit-while/lstm_cell_3533/split/split_dim:output:0%while/lstm_cell_3533/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџd:џџџџџџџџџd:џџџџџџџџџd:џџџџџџџџџd*
	num_split~
while/lstm_cell_3533/SigmoidSigmoid#while/lstm_cell_3533/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџd
while/lstm_cell_3533/Sigmoid_1Sigmoid#while/lstm_cell_3533/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџd
while/lstm_cell_3533/mulMul"while/lstm_cell_3533/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџdx
while/lstm_cell_3533/ReluRelu#while/lstm_cell_3533/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџd
while/lstm_cell_3533/mul_1Mul while/lstm_cell_3533/Sigmoid:y:0'while/lstm_cell_3533/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџd
while/lstm_cell_3533/add_1AddV2while/lstm_cell_3533/mul:z:0while/lstm_cell_3533/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџd
while/lstm_cell_3533/Sigmoid_2Sigmoid#while/lstm_cell_3533/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџdu
while/lstm_cell_3533/Relu_1Reluwhile/lstm_cell_3533/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџdЂ
while/lstm_cell_3533/mul_2Mul"while/lstm_cell_3533/Sigmoid_2:y:0)while/lstm_cell_3533/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџdr
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : я
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell_3533/mul_2:z:0*
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
: {
while/Identity_4Identitywhile/lstm_cell_3533/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџd{
while/Identity_5Identitywhile/lstm_cell_3533/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџdж

while/NoOpNoOp,^while/lstm_cell_3533/BiasAdd/ReadVariableOp+^while/lstm_cell_3533/MatMul/ReadVariableOp-^while/lstm_cell_3533/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"n
4while_lstm_cell_3533_biasadd_readvariableop_resource6while_lstm_cell_3533_biasadd_readvariableop_resource_0"p
5while_lstm_cell_3533_matmul_1_readvariableop_resource7while_lstm_cell_3533_matmul_1_readvariableop_resource_0"l
3while_lstm_cell_3533_matmul_readvariableop_resource5while_lstm_cell_3533_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџd:џџџџџџџџџd: : : : : 2Z
+while/lstm_cell_3533/BiasAdd/ReadVariableOp+while/lstm_cell_3533/BiasAdd/ReadVariableOp2X
*while/lstm_cell_3533/MatMul/ReadVariableOp*while/lstm_cell_3533/MatMul/ReadVariableOp2\
,while/lstm_cell_3533/MatMul_1/ReadVariableOp,while/lstm_cell_3533/MatMul_1/ReadVariableOp: 
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
д$
џ
while_body_24306920
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_02
while_lstm_cell_3533_24306944_0:	2
while_lstm_cell_3533_24306946_0:	d.
while_lstm_cell_3533_24306948_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor0
while_lstm_cell_3533_24306944:	0
while_lstm_cell_3533_24306946:	d,
while_lstm_cell_3533_24306948:	Ђ,while/lstm_cell_3533/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   І
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype0Х
,while/lstm_cell_3533/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_3533_24306944_0while_lstm_cell_3533_24306946_0while_lstm_cell_3533_24306948_0*
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
GPU 2J 8 *U
fPRN
L__inference_lstm_cell_3533_layer_call_and_return_conditional_losses_24306860r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : 
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:05while/lstm_cell_3533/StatefulPartitionedCall:output:0*
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
: 
while/Identity_4Identity5while/lstm_cell_3533/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџd
while/Identity_5Identity5while/lstm_cell_3533/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџd{

while/NoOpNoOp-^while/lstm_cell_3533/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"@
while_lstm_cell_3533_24306944while_lstm_cell_3533_24306944_0"@
while_lstm_cell_3533_24306946while_lstm_cell_3533_24306946_0"@
while_lstm_cell_3533_24306948while_lstm_cell_3533_24306948_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџd:џџџџџџџџџd: : : : : 2\
,while/lstm_cell_3533/StatefulPartitionedCall,while/lstm_cell_3533/StatefulPartitionedCall: 
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
Д:
ф
while_body_24307065
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0H
5while_lstm_cell_3533_matmul_readvariableop_resource_0:	J
7while_lstm_cell_3533_matmul_1_readvariableop_resource_0:	dE
6while_lstm_cell_3533_biasadd_readvariableop_resource_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorF
3while_lstm_cell_3533_matmul_readvariableop_resource:	H
5while_lstm_cell_3533_matmul_1_readvariableop_resource:	dC
4while_lstm_cell_3533_biasadd_readvariableop_resource:	Ђ+while/lstm_cell_3533/BiasAdd/ReadVariableOpЂ*while/lstm_cell_3533/MatMul/ReadVariableOpЂ,while/lstm_cell_3533/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   І
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype0Ё
*while/lstm_cell_3533/MatMul/ReadVariableOpReadVariableOp5while_lstm_cell_3533_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype0О
while/lstm_cell_3533/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:02while/lstm_cell_3533/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЅ
,while/lstm_cell_3533/MatMul_1/ReadVariableOpReadVariableOp7while_lstm_cell_3533_matmul_1_readvariableop_resource_0*
_output_shapes
:	d*
dtype0Ѕ
while/lstm_cell_3533/MatMul_1MatMulwhile_placeholder_24while/lstm_cell_3533/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЄ
while/lstm_cell_3533/addAddV2%while/lstm_cell_3533/MatMul:product:0'while/lstm_cell_3533/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџ
+while/lstm_cell_3533/BiasAdd/ReadVariableOpReadVariableOp6while_lstm_cell_3533_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0­
while/lstm_cell_3533/BiasAddBiasAddwhile/lstm_cell_3533/add:z:03while/lstm_cell_3533/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџf
$while/lstm_cell_3533/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ѕ
while/lstm_cell_3533/splitSplit-while/lstm_cell_3533/split/split_dim:output:0%while/lstm_cell_3533/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџd:џџџџџџџџџd:џџџџџџџџџd:џџџџџџџџџd*
	num_split~
while/lstm_cell_3533/SigmoidSigmoid#while/lstm_cell_3533/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџd
while/lstm_cell_3533/Sigmoid_1Sigmoid#while/lstm_cell_3533/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџd
while/lstm_cell_3533/mulMul"while/lstm_cell_3533/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџdx
while/lstm_cell_3533/ReluRelu#while/lstm_cell_3533/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџd
while/lstm_cell_3533/mul_1Mul while/lstm_cell_3533/Sigmoid:y:0'while/lstm_cell_3533/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџd
while/lstm_cell_3533/add_1AddV2while/lstm_cell_3533/mul:z:0while/lstm_cell_3533/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџd
while/lstm_cell_3533/Sigmoid_2Sigmoid#while/lstm_cell_3533/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџdu
while/lstm_cell_3533/Relu_1Reluwhile/lstm_cell_3533/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџdЂ
while/lstm_cell_3533/mul_2Mul"while/lstm_cell_3533/Sigmoid_2:y:0)while/lstm_cell_3533/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџdr
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : я
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell_3533/mul_2:z:0*
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
: {
while/Identity_4Identitywhile/lstm_cell_3533/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџd{
while/Identity_5Identitywhile/lstm_cell_3533/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџdж

while/NoOpNoOp,^while/lstm_cell_3533/BiasAdd/ReadVariableOp+^while/lstm_cell_3533/MatMul/ReadVariableOp-^while/lstm_cell_3533/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"n
4while_lstm_cell_3533_biasadd_readvariableop_resource6while_lstm_cell_3533_biasadd_readvariableop_resource_0"p
5while_lstm_cell_3533_matmul_1_readvariableop_resource7while_lstm_cell_3533_matmul_1_readvariableop_resource_0"l
3while_lstm_cell_3533_matmul_readvariableop_resource5while_lstm_cell_3533_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџd:џџџџџџџџџd: : : : : 2Z
+while/lstm_cell_3533/BiasAdd/ReadVariableOp+while/lstm_cell_3533/BiasAdd/ReadVariableOp2X
*while/lstm_cell_3533/MatMul/ReadVariableOp*while/lstm_cell_3533/MatMul/ReadVariableOp2\
,while/lstm_cell_3533/MatMul_1/ReadVariableOp,while/lstm_cell_3533/MatMul_1/ReadVariableOp: 
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
Д9

D__inference_lstm_5_layer_call_and_return_conditional_losses_24306797

inputs*
lstm_cell_3533_24306713:	*
lstm_cell_3533_24306715:	d&
lstm_cell_3533_24306717:	
identityЂ&lstm_cell_3533/StatefulPartitionedCallЂwhile;
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
shrink_axis_mask
&lstm_cell_3533/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_3533_24306713lstm_cell_3533_24306715lstm_cell_3533_24306717*
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
GPU 2J 8 *U
fPRN
L__inference_lstm_cell_3533_layer_call_and_return_conditional_losses_24306712n
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
value	B : Ч
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_3533_24306713lstm_cell_3533_24306715lstm_cell_3533_24306717*
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
bodyR
while_body_24306727*
condR
while_cond_24306726*K
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
:џџџџџџџџџdw
NoOpNoOp'^lstm_cell_3533/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџџџџџџџџџџ: : : 2P
&lstm_cell_3533/StatefulPartitionedCall&lstm_cell_3533/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
П
Э
while_cond_24308519
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_24308519___redundant_placeholder06
2while_while_cond_24308519___redundant_placeholder16
2while_while_cond_24308519___redundant_placeholder26
2while_while_cond_24308519___redundant_placeholder3
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
Д:
ф
while_body_24308375
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0H
5while_lstm_cell_3533_matmul_readvariableop_resource_0:	J
7while_lstm_cell_3533_matmul_1_readvariableop_resource_0:	dE
6while_lstm_cell_3533_biasadd_readvariableop_resource_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorF
3while_lstm_cell_3533_matmul_readvariableop_resource:	H
5while_lstm_cell_3533_matmul_1_readvariableop_resource:	dC
4while_lstm_cell_3533_biasadd_readvariableop_resource:	Ђ+while/lstm_cell_3533/BiasAdd/ReadVariableOpЂ*while/lstm_cell_3533/MatMul/ReadVariableOpЂ,while/lstm_cell_3533/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   І
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype0Ё
*while/lstm_cell_3533/MatMul/ReadVariableOpReadVariableOp5while_lstm_cell_3533_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype0О
while/lstm_cell_3533/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:02while/lstm_cell_3533/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЅ
,while/lstm_cell_3533/MatMul_1/ReadVariableOpReadVariableOp7while_lstm_cell_3533_matmul_1_readvariableop_resource_0*
_output_shapes
:	d*
dtype0Ѕ
while/lstm_cell_3533/MatMul_1MatMulwhile_placeholder_24while/lstm_cell_3533/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЄ
while/lstm_cell_3533/addAddV2%while/lstm_cell_3533/MatMul:product:0'while/lstm_cell_3533/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџ
+while/lstm_cell_3533/BiasAdd/ReadVariableOpReadVariableOp6while_lstm_cell_3533_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0­
while/lstm_cell_3533/BiasAddBiasAddwhile/lstm_cell_3533/add:z:03while/lstm_cell_3533/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџf
$while/lstm_cell_3533/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ѕ
while/lstm_cell_3533/splitSplit-while/lstm_cell_3533/split/split_dim:output:0%while/lstm_cell_3533/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџd:џџџџџџџџџd:џџџџџџџџџd:џџџџџџџџџd*
	num_split~
while/lstm_cell_3533/SigmoidSigmoid#while/lstm_cell_3533/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџd
while/lstm_cell_3533/Sigmoid_1Sigmoid#while/lstm_cell_3533/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџd
while/lstm_cell_3533/mulMul"while/lstm_cell_3533/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџdx
while/lstm_cell_3533/ReluRelu#while/lstm_cell_3533/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџd
while/lstm_cell_3533/mul_1Mul while/lstm_cell_3533/Sigmoid:y:0'while/lstm_cell_3533/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџd
while/lstm_cell_3533/add_1AddV2while/lstm_cell_3533/mul:z:0while/lstm_cell_3533/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџd
while/lstm_cell_3533/Sigmoid_2Sigmoid#while/lstm_cell_3533/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџdu
while/lstm_cell_3533/Relu_1Reluwhile/lstm_cell_3533/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџdЂ
while/lstm_cell_3533/mul_2Mul"while/lstm_cell_3533/Sigmoid_2:y:0)while/lstm_cell_3533/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџdr
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : я
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell_3533/mul_2:z:0*
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
: {
while/Identity_4Identitywhile/lstm_cell_3533/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџd{
while/Identity_5Identitywhile/lstm_cell_3533/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџdж

while/NoOpNoOp,^while/lstm_cell_3533/BiasAdd/ReadVariableOp+^while/lstm_cell_3533/MatMul/ReadVariableOp-^while/lstm_cell_3533/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"n
4while_lstm_cell_3533_biasadd_readvariableop_resource6while_lstm_cell_3533_biasadd_readvariableop_resource_0"p
5while_lstm_cell_3533_matmul_1_readvariableop_resource7while_lstm_cell_3533_matmul_1_readvariableop_resource_0"l
3while_lstm_cell_3533_matmul_readvariableop_resource5while_lstm_cell_3533_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџd:џџџџџџџџџd: : : : : 2Z
+while/lstm_cell_3533/BiasAdd/ReadVariableOp+while/lstm_cell_3533/BiasAdd/ReadVariableOp2X
*while/lstm_cell_3533/MatMul/ReadVariableOp*while/lstm_cell_3533/MatMul/ReadVariableOp2\
,while/lstm_cell_3533/MatMul_1/ReadVariableOp,while/lstm_cell_3533/MatMul_1/ReadVariableOp: 
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
Д9

D__inference_lstm_5_layer_call_and_return_conditional_losses_24306990

inputs*
lstm_cell_3533_24306906:	*
lstm_cell_3533_24306908:	d&
lstm_cell_3533_24306910:	
identityЂ&lstm_cell_3533/StatefulPartitionedCallЂwhile;
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
shrink_axis_mask
&lstm_cell_3533/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_3533_24306906lstm_cell_3533_24306908lstm_cell_3533_24306910*
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
GPU 2J 8 *U
fPRN
L__inference_lstm_cell_3533_layer_call_and_return_conditional_losses_24306860n
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
value	B : Ч
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_3533_24306906lstm_cell_3533_24306908lstm_cell_3533_24306910*
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
bodyR
while_body_24306920*
condR
while_cond_24306919*K
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
:џџџџџџџџџdw
NoOpNoOp'^lstm_cell_3533/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџџџџџџџџџџ: : : 2P
&lstm_cell_3533/StatefulPartitionedCall&lstm_cell_3533/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs


о
/__inference_sequential_5_layer_call_fn_24307628

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
identityЂStatefulPartitionedCallК
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
GPU 2J 8 *S
fNRL
J__inference_sequential_5_layer_call_and_return_conditional_losses_24307209o
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
П
Э
while_cond_24306726
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_24306726___redundant_placeholder06
2while_while_cond_24306726___redundant_placeholder16
2while_while_cond_24306726___redundant_placeholder26
2while_while_cond_24306726___redundant_placeholder3
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
Щ	
ї
F__inference_dense_17_layer_call_and_return_conditional_losses_24307202

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
ђl

J__inference_sequential_5_layer_call_and_return_conditional_losses_24307816

inputsG
4lstm_5_lstm_cell_3533_matmul_readvariableop_resource:	I
6lstm_5_lstm_cell_3533_matmul_1_readvariableop_resource:	dD
5lstm_5_lstm_cell_3533_biasadd_readvariableop_resource:	9
'dense_15_matmul_readvariableop_resource:d26
(dense_15_biasadd_readvariableop_resource:29
'dense_16_matmul_readvariableop_resource:226
(dense_16_biasadd_readvariableop_resource:29
'dense_17_matmul_readvariableop_resource:26
(dense_17_biasadd_readvariableop_resource:
identityЂdense_15/BiasAdd/ReadVariableOpЂdense_15/MatMul/ReadVariableOpЂdense_16/BiasAdd/ReadVariableOpЂdense_16/MatMul/ReadVariableOpЂdense_17/BiasAdd/ReadVariableOpЂdense_17/MatMul/ReadVariableOpЂ,lstm_5/lstm_cell_3533/BiasAdd/ReadVariableOpЂ+lstm_5/lstm_cell_3533/MatMul/ReadVariableOpЂ-lstm_5/lstm_cell_3533/MatMul_1/ReadVariableOpЂlstm_5/whileB
lstm_5/ShapeShapeinputs*
T0*
_output_shapes
:d
lstm_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: f
lstm_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:f
lstm_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:є
lstm_5/strided_sliceStridedSlicelstm_5/Shape:output:0#lstm_5/strided_slice/stack:output:0%lstm_5/strided_slice/stack_1:output:0%lstm_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskW
lstm_5/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d
lstm_5/zeros/packedPacklstm_5/strided_slice:output:0lstm_5/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:W
lstm_5/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
lstm_5/zerosFilllstm_5/zeros/packed:output:0lstm_5/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџdY
lstm_5/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d
lstm_5/zeros_1/packedPacklstm_5/strided_slice:output:0 lstm_5/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:Y
lstm_5/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
lstm_5/zeros_1Filllstm_5/zeros_1/packed:output:0lstm_5/zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџdj
lstm_5/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          {
lstm_5/transpose	Transposeinputslstm_5/transpose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџR
lstm_5/Shape_1Shapelstm_5/transpose:y:0*
T0*
_output_shapes
:f
lstm_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: h
lstm_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
lstm_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ў
lstm_5/strided_slice_1StridedSlicelstm_5/Shape_1:output:0%lstm_5/strided_slice_1/stack:output:0'lstm_5/strided_slice_1/stack_1:output:0'lstm_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskm
"lstm_5/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџЩ
lstm_5/TensorArrayV2TensorListReserve+lstm_5/TensorArrayV2/element_shape:output:0lstm_5/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
<lstm_5/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   ѕ
.lstm_5/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_5/transpose:y:0Elstm_5/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвf
lstm_5/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: h
lstm_5/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
lstm_5/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
lstm_5/strided_slice_2StridedSlicelstm_5/transpose:y:0%lstm_5/strided_slice_2/stack:output:0'lstm_5/strided_slice_2/stack_1:output:0'lstm_5/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_maskЁ
+lstm_5/lstm_cell_3533/MatMul/ReadVariableOpReadVariableOp4lstm_5_lstm_cell_3533_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0Џ
lstm_5/lstm_cell_3533/MatMulMatMullstm_5/strided_slice_2:output:03lstm_5/lstm_cell_3533/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЅ
-lstm_5/lstm_cell_3533/MatMul_1/ReadVariableOpReadVariableOp6lstm_5_lstm_cell_3533_matmul_1_readvariableop_resource*
_output_shapes
:	d*
dtype0Љ
lstm_5/lstm_cell_3533/MatMul_1MatMullstm_5/zeros:output:05lstm_5/lstm_cell_3533/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЇ
lstm_5/lstm_cell_3533/addAddV2&lstm_5/lstm_cell_3533/MatMul:product:0(lstm_5/lstm_cell_3533/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџ
,lstm_5/lstm_cell_3533/BiasAdd/ReadVariableOpReadVariableOp5lstm_5_lstm_cell_3533_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0А
lstm_5/lstm_cell_3533/BiasAddBiasAddlstm_5/lstm_cell_3533/add:z:04lstm_5/lstm_cell_3533/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџg
%lstm_5/lstm_cell_3533/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ј
lstm_5/lstm_cell_3533/splitSplit.lstm_5/lstm_cell_3533/split/split_dim:output:0&lstm_5/lstm_cell_3533/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџd:џџџџџџџџџd:џџџџџџџџџd:џџџџџџџџџd*
	num_split
lstm_5/lstm_cell_3533/SigmoidSigmoid$lstm_5/lstm_cell_3533/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџd
lstm_5/lstm_cell_3533/Sigmoid_1Sigmoid$lstm_5/lstm_cell_3533/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџd
lstm_5/lstm_cell_3533/mulMul#lstm_5/lstm_cell_3533/Sigmoid_1:y:0lstm_5/zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџdz
lstm_5/lstm_cell_3533/ReluRelu$lstm_5/lstm_cell_3533/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџdЁ
lstm_5/lstm_cell_3533/mul_1Mul!lstm_5/lstm_cell_3533/Sigmoid:y:0(lstm_5/lstm_cell_3533/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџd
lstm_5/lstm_cell_3533/add_1AddV2lstm_5/lstm_cell_3533/mul:z:0lstm_5/lstm_cell_3533/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџd
lstm_5/lstm_cell_3533/Sigmoid_2Sigmoid$lstm_5/lstm_cell_3533/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџdw
lstm_5/lstm_cell_3533/Relu_1Relulstm_5/lstm_cell_3533/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџdЅ
lstm_5/lstm_cell_3533/mul_2Mul#lstm_5/lstm_cell_3533/Sigmoid_2:y:0*lstm_5/lstm_cell_3533/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџdu
$lstm_5/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџd   e
#lstm_5/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :к
lstm_5/TensorArrayV2_1TensorListReserve-lstm_5/TensorArrayV2_1/element_shape:output:0,lstm_5/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвM
lstm_5/timeConst*
_output_shapes
: *
dtype0*
value	B : j
lstm_5/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ[
lstm_5/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ю
lstm_5/whileWhile"lstm_5/while/loop_counter:output:0(lstm_5/while/maximum_iterations:output:0lstm_5/time:output:0lstm_5/TensorArrayV2_1:handle:0lstm_5/zeros:output:0lstm_5/zeros_1:output:0lstm_5/strided_slice_1:output:0>lstm_5/TensorArrayUnstack/TensorListFromTensor:output_handle:04lstm_5_lstm_cell_3533_matmul_readvariableop_resource6lstm_5_lstm_cell_3533_matmul_1_readvariableop_resource5lstm_5_lstm_cell_3533_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџd:џџџџџџџџџd: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *&
bodyR
lstm_5_while_body_24307711*&
condR
lstm_5_while_cond_24307710*K
output_shapes:
8: : : : :џџџџџџџџџd:џџџџџџџџџd: : : : : *
parallel_iterations 
7lstm_5/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџd   ы
)lstm_5/TensorArrayV2Stack/TensorListStackTensorListStacklstm_5/while:output:3@lstm_5/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџd*
element_dtype0*
num_elementso
lstm_5/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџh
lstm_5/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: h
lstm_5/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Њ
lstm_5/strided_slice_3StridedSlice2lstm_5/TensorArrayV2Stack/TensorListStack:tensor:0%lstm_5/strided_slice_3/stack:output:0'lstm_5/strided_slice_3/stack_1:output:0'lstm_5/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџd*
shrink_axis_maskl
lstm_5/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ћ
lstm_5/transpose_1	Transpose2lstm_5/TensorArrayV2Stack/TensorListStack:tensor:0 lstm_5/transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџdb
lstm_5/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    
dense_15/MatMul/ReadVariableOpReadVariableOp'dense_15_matmul_readvariableop_resource*
_output_shapes

:d2*
dtype0
dense_15/MatMulMatMullstm_5/strided_slice_3:output:0&dense_15/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_15/BiasAdd/ReadVariableOpReadVariableOp(dense_15_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0
dense_15/BiasAddBiasAdddense_15/MatMul:product:0'dense_15/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2b
dense_15/ReluReludense_15/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_16/MatMul/ReadVariableOpReadVariableOp'dense_16_matmul_readvariableop_resource*
_output_shapes

:22*
dtype0
dense_16/MatMulMatMuldense_15/Relu:activations:0&dense_16/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_16/BiasAdd/ReadVariableOpReadVariableOp(dense_16_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0
dense_16/BiasAddBiasAdddense_16/MatMul:product:0'dense_16/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2b
dense_16/ReluReludense_16/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_17/MatMul/ReadVariableOpReadVariableOp'dense_17_matmul_readvariableop_resource*
_output_shapes

:2*
dtype0
dense_17/MatMulMatMuldense_16/Relu:activations:0&dense_17/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
dense_17/BiasAdd/ReadVariableOpReadVariableOp(dense_17_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_17/BiasAddBiasAdddense_17/MatMul:product:0'dense_17/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџh
IdentityIdentitydense_17/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџЋ
NoOpNoOp ^dense_15/BiasAdd/ReadVariableOp^dense_15/MatMul/ReadVariableOp ^dense_16/BiasAdd/ReadVariableOp^dense_16/MatMul/ReadVariableOp ^dense_17/BiasAdd/ReadVariableOp^dense_17/MatMul/ReadVariableOp-^lstm_5/lstm_cell_3533/BiasAdd/ReadVariableOp,^lstm_5/lstm_cell_3533/MatMul/ReadVariableOp.^lstm_5/lstm_cell_3533/MatMul_1/ReadVariableOp^lstm_5/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):џџџџџџџџџ: : : : : : : : : 2B
dense_15/BiasAdd/ReadVariableOpdense_15/BiasAdd/ReadVariableOp2@
dense_15/MatMul/ReadVariableOpdense_15/MatMul/ReadVariableOp2B
dense_16/BiasAdd/ReadVariableOpdense_16/BiasAdd/ReadVariableOp2@
dense_16/MatMul/ReadVariableOpdense_16/MatMul/ReadVariableOp2B
dense_17/BiasAdd/ReadVariableOpdense_17/BiasAdd/ReadVariableOp2@
dense_17/MatMul/ReadVariableOpdense_17/MatMul/ReadVariableOp2\
,lstm_5/lstm_cell_3533/BiasAdd/ReadVariableOp,lstm_5/lstm_cell_3533/BiasAdd/ReadVariableOp2Z
+lstm_5/lstm_cell_3533/MatMul/ReadVariableOp+lstm_5/lstm_cell_3533/MatMul/ReadVariableOp2^
-lstm_5/lstm_cell_3533/MatMul_1/ReadVariableOp-lstm_5/lstm_cell_3533/MatMul_1/ReadVariableOp2
lstm_5/whilelstm_5/while:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs


ї
F__inference_dense_16_layer_call_and_return_conditional_losses_24307186

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
 
з	
#__inference__wrapped_model_24306645
lstm_5_inputT
Asequential_5_lstm_5_lstm_cell_3533_matmul_readvariableop_resource:	V
Csequential_5_lstm_5_lstm_cell_3533_matmul_1_readvariableop_resource:	dQ
Bsequential_5_lstm_5_lstm_cell_3533_biasadd_readvariableop_resource:	F
4sequential_5_dense_15_matmul_readvariableop_resource:d2C
5sequential_5_dense_15_biasadd_readvariableop_resource:2F
4sequential_5_dense_16_matmul_readvariableop_resource:22C
5sequential_5_dense_16_biasadd_readvariableop_resource:2F
4sequential_5_dense_17_matmul_readvariableop_resource:2C
5sequential_5_dense_17_biasadd_readvariableop_resource:
identityЂ,sequential_5/dense_15/BiasAdd/ReadVariableOpЂ+sequential_5/dense_15/MatMul/ReadVariableOpЂ,sequential_5/dense_16/BiasAdd/ReadVariableOpЂ+sequential_5/dense_16/MatMul/ReadVariableOpЂ,sequential_5/dense_17/BiasAdd/ReadVariableOpЂ+sequential_5/dense_17/MatMul/ReadVariableOpЂ9sequential_5/lstm_5/lstm_cell_3533/BiasAdd/ReadVariableOpЂ8sequential_5/lstm_5/lstm_cell_3533/MatMul/ReadVariableOpЂ:sequential_5/lstm_5/lstm_cell_3533/MatMul_1/ReadVariableOpЂsequential_5/lstm_5/whileU
sequential_5/lstm_5/ShapeShapelstm_5_input*
T0*
_output_shapes
:q
'sequential_5/lstm_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)sequential_5/lstm_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)sequential_5/lstm_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Е
!sequential_5/lstm_5/strided_sliceStridedSlice"sequential_5/lstm_5/Shape:output:00sequential_5/lstm_5/strided_slice/stack:output:02sequential_5/lstm_5/strided_slice/stack_1:output:02sequential_5/lstm_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"sequential_5/lstm_5/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :dЏ
 sequential_5/lstm_5/zeros/packedPack*sequential_5/lstm_5/strided_slice:output:0+sequential_5/lstm_5/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:d
sequential_5/lstm_5/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Ј
sequential_5/lstm_5/zerosFill)sequential_5/lstm_5/zeros/packed:output:0(sequential_5/lstm_5/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџdf
$sequential_5/lstm_5/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :dГ
"sequential_5/lstm_5/zeros_1/packedPack*sequential_5/lstm_5/strided_slice:output:0-sequential_5/lstm_5/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:f
!sequential_5/lstm_5/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Ў
sequential_5/lstm_5/zeros_1Fill+sequential_5/lstm_5/zeros_1/packed:output:0*sequential_5/lstm_5/zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџdw
"sequential_5/lstm_5/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
sequential_5/lstm_5/transpose	Transposelstm_5_input+sequential_5/lstm_5/transpose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџl
sequential_5/lstm_5/Shape_1Shape!sequential_5/lstm_5/transpose:y:0*
T0*
_output_shapes
:s
)sequential_5/lstm_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+sequential_5/lstm_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+sequential_5/lstm_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:П
#sequential_5/lstm_5/strided_slice_1StridedSlice$sequential_5/lstm_5/Shape_1:output:02sequential_5/lstm_5/strided_slice_1/stack:output:04sequential_5/lstm_5/strided_slice_1/stack_1:output:04sequential_5/lstm_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskz
/sequential_5/lstm_5/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ№
!sequential_5/lstm_5/TensorArrayV2TensorListReserve8sequential_5/lstm_5/TensorArrayV2/element_shape:output:0,sequential_5/lstm_5/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
Isequential_5/lstm_5/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   
;sequential_5/lstm_5/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor!sequential_5/lstm_5/transpose:y:0Rsequential_5/lstm_5/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвs
)sequential_5/lstm_5/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+sequential_5/lstm_5/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+sequential_5/lstm_5/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Э
#sequential_5/lstm_5/strided_slice_2StridedSlice!sequential_5/lstm_5/transpose:y:02sequential_5/lstm_5/strided_slice_2/stack:output:04sequential_5/lstm_5/strided_slice_2/stack_1:output:04sequential_5/lstm_5/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_maskЛ
8sequential_5/lstm_5/lstm_cell_3533/MatMul/ReadVariableOpReadVariableOpAsequential_5_lstm_5_lstm_cell_3533_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0ж
)sequential_5/lstm_5/lstm_cell_3533/MatMulMatMul,sequential_5/lstm_5/strided_slice_2:output:0@sequential_5/lstm_5/lstm_cell_3533/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџП
:sequential_5/lstm_5/lstm_cell_3533/MatMul_1/ReadVariableOpReadVariableOpCsequential_5_lstm_5_lstm_cell_3533_matmul_1_readvariableop_resource*
_output_shapes
:	d*
dtype0а
+sequential_5/lstm_5/lstm_cell_3533/MatMul_1MatMul"sequential_5/lstm_5/zeros:output:0Bsequential_5/lstm_5/lstm_cell_3533/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЮ
&sequential_5/lstm_5/lstm_cell_3533/addAddV23sequential_5/lstm_5/lstm_cell_3533/MatMul:product:05sequential_5/lstm_5/lstm_cell_3533/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџЙ
9sequential_5/lstm_5/lstm_cell_3533/BiasAdd/ReadVariableOpReadVariableOpBsequential_5_lstm_5_lstm_cell_3533_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0з
*sequential_5/lstm_5/lstm_cell_3533/BiasAddBiasAdd*sequential_5/lstm_5/lstm_cell_3533/add:z:0Asequential_5/lstm_5/lstm_cell_3533/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџt
2sequential_5/lstm_5/lstm_cell_3533/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
(sequential_5/lstm_5/lstm_cell_3533/splitSplit;sequential_5/lstm_5/lstm_cell_3533/split/split_dim:output:03sequential_5/lstm_5/lstm_cell_3533/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџd:џџџџџџџџџd:џџџџџџџџџd:џџџџџџџџџd*
	num_split
*sequential_5/lstm_5/lstm_cell_3533/SigmoidSigmoid1sequential_5/lstm_5/lstm_cell_3533/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџd
,sequential_5/lstm_5/lstm_cell_3533/Sigmoid_1Sigmoid1sequential_5/lstm_5/lstm_cell_3533/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџdЗ
&sequential_5/lstm_5/lstm_cell_3533/mulMul0sequential_5/lstm_5/lstm_cell_3533/Sigmoid_1:y:0$sequential_5/lstm_5/zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџd
'sequential_5/lstm_5/lstm_cell_3533/ReluRelu1sequential_5/lstm_5/lstm_cell_3533/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџdШ
(sequential_5/lstm_5/lstm_cell_3533/mul_1Mul.sequential_5/lstm_5/lstm_cell_3533/Sigmoid:y:05sequential_5/lstm_5/lstm_cell_3533/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџdН
(sequential_5/lstm_5/lstm_cell_3533/add_1AddV2*sequential_5/lstm_5/lstm_cell_3533/mul:z:0,sequential_5/lstm_5/lstm_cell_3533/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџd
,sequential_5/lstm_5/lstm_cell_3533/Sigmoid_2Sigmoid1sequential_5/lstm_5/lstm_cell_3533/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџd
)sequential_5/lstm_5/lstm_cell_3533/Relu_1Relu,sequential_5/lstm_5/lstm_cell_3533/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџdЬ
(sequential_5/lstm_5/lstm_cell_3533/mul_2Mul0sequential_5/lstm_5/lstm_cell_3533/Sigmoid_2:y:07sequential_5/lstm_5/lstm_cell_3533/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџd
1sequential_5/lstm_5/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџd   r
0sequential_5/lstm_5/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :
#sequential_5/lstm_5/TensorArrayV2_1TensorListReserve:sequential_5/lstm_5/TensorArrayV2_1/element_shape:output:09sequential_5/lstm_5/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвZ
sequential_5/lstm_5/timeConst*
_output_shapes
: *
dtype0*
value	B : w
,sequential_5/lstm_5/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџh
&sequential_5/lstm_5/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Є
sequential_5/lstm_5/whileWhile/sequential_5/lstm_5/while/loop_counter:output:05sequential_5/lstm_5/while/maximum_iterations:output:0!sequential_5/lstm_5/time:output:0,sequential_5/lstm_5/TensorArrayV2_1:handle:0"sequential_5/lstm_5/zeros:output:0$sequential_5/lstm_5/zeros_1:output:0,sequential_5/lstm_5/strided_slice_1:output:0Ksequential_5/lstm_5/TensorArrayUnstack/TensorListFromTensor:output_handle:0Asequential_5_lstm_5_lstm_cell_3533_matmul_readvariableop_resourceCsequential_5_lstm_5_lstm_cell_3533_matmul_1_readvariableop_resourceBsequential_5_lstm_5_lstm_cell_3533_biasadd_readvariableop_resource*
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
'sequential_5_lstm_5_while_body_24306540*3
cond+R)
'sequential_5_lstm_5_while_cond_24306539*K
output_shapes:
8: : : : :џџџџџџџџџd:џџџџџџџџџd: : : : : *
parallel_iterations 
Dsequential_5/lstm_5/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџd   
6sequential_5/lstm_5/TensorArrayV2Stack/TensorListStackTensorListStack"sequential_5/lstm_5/while:output:3Msequential_5/lstm_5/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџd*
element_dtype0*
num_elements|
)sequential_5/lstm_5/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџu
+sequential_5/lstm_5/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: u
+sequential_5/lstm_5/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ы
#sequential_5/lstm_5/strided_slice_3StridedSlice?sequential_5/lstm_5/TensorArrayV2Stack/TensorListStack:tensor:02sequential_5/lstm_5/strided_slice_3/stack:output:04sequential_5/lstm_5/strided_slice_3/stack_1:output:04sequential_5/lstm_5/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџd*
shrink_axis_masky
$sequential_5/lstm_5/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          в
sequential_5/lstm_5/transpose_1	Transpose?sequential_5/lstm_5/TensorArrayV2Stack/TensorListStack:tensor:0-sequential_5/lstm_5/transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџdo
sequential_5/lstm_5/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *     
+sequential_5/dense_15/MatMul/ReadVariableOpReadVariableOp4sequential_5_dense_15_matmul_readvariableop_resource*
_output_shapes

:d2*
dtype0Л
sequential_5/dense_15/MatMulMatMul,sequential_5/lstm_5/strided_slice_3:output:03sequential_5/dense_15/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
,sequential_5/dense_15/BiasAdd/ReadVariableOpReadVariableOp5sequential_5_dense_15_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0И
sequential_5/dense_15/BiasAddBiasAdd&sequential_5/dense_15/MatMul:product:04sequential_5/dense_15/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2|
sequential_5/dense_15/ReluRelu&sequential_5/dense_15/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2 
+sequential_5/dense_16/MatMul/ReadVariableOpReadVariableOp4sequential_5_dense_16_matmul_readvariableop_resource*
_output_shapes

:22*
dtype0З
sequential_5/dense_16/MatMulMatMul(sequential_5/dense_15/Relu:activations:03sequential_5/dense_16/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
,sequential_5/dense_16/BiasAdd/ReadVariableOpReadVariableOp5sequential_5_dense_16_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0И
sequential_5/dense_16/BiasAddBiasAdd&sequential_5/dense_16/MatMul:product:04sequential_5/dense_16/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2|
sequential_5/dense_16/ReluRelu&sequential_5/dense_16/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2 
+sequential_5/dense_17/MatMul/ReadVariableOpReadVariableOp4sequential_5_dense_17_matmul_readvariableop_resource*
_output_shapes

:2*
dtype0З
sequential_5/dense_17/MatMulMatMul(sequential_5/dense_16/Relu:activations:03sequential_5/dense_17/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
,sequential_5/dense_17/BiasAdd/ReadVariableOpReadVariableOp5sequential_5_dense_17_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0И
sequential_5/dense_17/BiasAddBiasAdd&sequential_5/dense_17/MatMul:product:04sequential_5/dense_17/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџu
IdentityIdentity&sequential_5/dense_17/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ­
NoOpNoOp-^sequential_5/dense_15/BiasAdd/ReadVariableOp,^sequential_5/dense_15/MatMul/ReadVariableOp-^sequential_5/dense_16/BiasAdd/ReadVariableOp,^sequential_5/dense_16/MatMul/ReadVariableOp-^sequential_5/dense_17/BiasAdd/ReadVariableOp,^sequential_5/dense_17/MatMul/ReadVariableOp:^sequential_5/lstm_5/lstm_cell_3533/BiasAdd/ReadVariableOp9^sequential_5/lstm_5/lstm_cell_3533/MatMul/ReadVariableOp;^sequential_5/lstm_5/lstm_cell_3533/MatMul_1/ReadVariableOp^sequential_5/lstm_5/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):џџџџџџџџџ: : : : : : : : : 2\
,sequential_5/dense_15/BiasAdd/ReadVariableOp,sequential_5/dense_15/BiasAdd/ReadVariableOp2Z
+sequential_5/dense_15/MatMul/ReadVariableOp+sequential_5/dense_15/MatMul/ReadVariableOp2\
,sequential_5/dense_16/BiasAdd/ReadVariableOp,sequential_5/dense_16/BiasAdd/ReadVariableOp2Z
+sequential_5/dense_16/MatMul/ReadVariableOp+sequential_5/dense_16/MatMul/ReadVariableOp2\
,sequential_5/dense_17/BiasAdd/ReadVariableOp,sequential_5/dense_17/BiasAdd/ReadVariableOp2Z
+sequential_5/dense_17/MatMul/ReadVariableOp+sequential_5/dense_17/MatMul/ReadVariableOp2v
9sequential_5/lstm_5/lstm_cell_3533/BiasAdd/ReadVariableOp9sequential_5/lstm_5/lstm_cell_3533/BiasAdd/ReadVariableOp2t
8sequential_5/lstm_5/lstm_cell_3533/MatMul/ReadVariableOp8sequential_5/lstm_5/lstm_cell_3533/MatMul/ReadVariableOp2x
:sequential_5/lstm_5/lstm_cell_3533/MatMul_1/ReadVariableOp:sequential_5/lstm_5/lstm_cell_3533/MatMul_1/ReadVariableOp26
sequential_5/lstm_5/whilesequential_5/lstm_5/while:Y U
+
_output_shapes
:џџџџџџџџџ
&
_user_specified_namelstm_5_input
П
Э
while_cond_24306919
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_24306919___redundant_placeholder06
2while_while_cond_24306919___redundant_placeholder16
2while_while_cond_24306919___redundant_placeholder26
2while_while_cond_24306919___redundant_placeholder3
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
у	
л
&__inference_signature_wrapper_24307605
lstm_5_input
unknown:	
	unknown_0:	d
	unknown_1:	
	unknown_2:d2
	unknown_3:2
	unknown_4:22
	unknown_5:2
	unknown_6:2
	unknown_7:
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCalllstm_5_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
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
GPU 2J 8 *,
f'R%
#__inference__wrapped_model_24306645o
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
StatefulPartitionedCallStatefulPartitionedCall:Y U
+
_output_shapes
:џџџџџџџџџ
&
_user_specified_namelstm_5_input
Ц

+__inference_dense_16_layer_call_fn_24308634

inputs
unknown:22
	unknown_0:2
identityЂStatefulPartitionedCallл
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
GPU 2J 8 *O
fJRH
F__inference_dense_16_layer_call_and_return_conditional_losses_24307186o
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

И
)__inference_lstm_5_layer_call_fn_24308003
inputs_0
unknown:	
	unknown_0:	d
	unknown_1:	
identityЂStatefulPartitionedCallш
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
GPU 2J 8 *M
fHRF
D__inference_lstm_5_layer_call_and_return_conditional_losses_24306990o
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


ї
F__inference_dense_15_layer_call_and_return_conditional_losses_24308625

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
мB
Ф

lstm_5_while_body_24307876*
&lstm_5_while_lstm_5_while_loop_counter0
,lstm_5_while_lstm_5_while_maximum_iterations
lstm_5_while_placeholder
lstm_5_while_placeholder_1
lstm_5_while_placeholder_2
lstm_5_while_placeholder_3)
%lstm_5_while_lstm_5_strided_slice_1_0e
alstm_5_while_tensorarrayv2read_tensorlistgetitem_lstm_5_tensorarrayunstack_tensorlistfromtensor_0O
<lstm_5_while_lstm_cell_3533_matmul_readvariableop_resource_0:	Q
>lstm_5_while_lstm_cell_3533_matmul_1_readvariableop_resource_0:	dL
=lstm_5_while_lstm_cell_3533_biasadd_readvariableop_resource_0:	
lstm_5_while_identity
lstm_5_while_identity_1
lstm_5_while_identity_2
lstm_5_while_identity_3
lstm_5_while_identity_4
lstm_5_while_identity_5'
#lstm_5_while_lstm_5_strided_slice_1c
_lstm_5_while_tensorarrayv2read_tensorlistgetitem_lstm_5_tensorarrayunstack_tensorlistfromtensorM
:lstm_5_while_lstm_cell_3533_matmul_readvariableop_resource:	O
<lstm_5_while_lstm_cell_3533_matmul_1_readvariableop_resource:	dJ
;lstm_5_while_lstm_cell_3533_biasadd_readvariableop_resource:	Ђ2lstm_5/while/lstm_cell_3533/BiasAdd/ReadVariableOpЂ1lstm_5/while/lstm_cell_3533/MatMul/ReadVariableOpЂ3lstm_5/while/lstm_cell_3533/MatMul_1/ReadVariableOp
>lstm_5/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   Щ
0lstm_5/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemalstm_5_while_tensorarrayv2read_tensorlistgetitem_lstm_5_tensorarrayunstack_tensorlistfromtensor_0lstm_5_while_placeholderGlstm_5/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype0Џ
1lstm_5/while/lstm_cell_3533/MatMul/ReadVariableOpReadVariableOp<lstm_5_while_lstm_cell_3533_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype0г
"lstm_5/while/lstm_cell_3533/MatMulMatMul7lstm_5/while/TensorArrayV2Read/TensorListGetItem:item:09lstm_5/while/lstm_cell_3533/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџГ
3lstm_5/while/lstm_cell_3533/MatMul_1/ReadVariableOpReadVariableOp>lstm_5_while_lstm_cell_3533_matmul_1_readvariableop_resource_0*
_output_shapes
:	d*
dtype0К
$lstm_5/while/lstm_cell_3533/MatMul_1MatMullstm_5_while_placeholder_2;lstm_5/while/lstm_cell_3533/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЙ
lstm_5/while/lstm_cell_3533/addAddV2,lstm_5/while/lstm_cell_3533/MatMul:product:0.lstm_5/while/lstm_cell_3533/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџ­
2lstm_5/while/lstm_cell_3533/BiasAdd/ReadVariableOpReadVariableOp=lstm_5_while_lstm_cell_3533_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0Т
#lstm_5/while/lstm_cell_3533/BiasAddBiasAdd#lstm_5/while/lstm_cell_3533/add:z:0:lstm_5/while/lstm_cell_3533/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџm
+lstm_5/while/lstm_cell_3533/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
!lstm_5/while/lstm_cell_3533/splitSplit4lstm_5/while/lstm_cell_3533/split/split_dim:output:0,lstm_5/while/lstm_cell_3533/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџd:џџџџџџџџџd:џџџџџџџџџd:џџџџџџџџџd*
	num_split
#lstm_5/while/lstm_cell_3533/SigmoidSigmoid*lstm_5/while/lstm_cell_3533/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџd
%lstm_5/while/lstm_cell_3533/Sigmoid_1Sigmoid*lstm_5/while/lstm_cell_3533/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџd
lstm_5/while/lstm_cell_3533/mulMul)lstm_5/while/lstm_cell_3533/Sigmoid_1:y:0lstm_5_while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџd
 lstm_5/while/lstm_cell_3533/ReluRelu*lstm_5/while/lstm_cell_3533/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџdГ
!lstm_5/while/lstm_cell_3533/mul_1Mul'lstm_5/while/lstm_cell_3533/Sigmoid:y:0.lstm_5/while/lstm_cell_3533/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџdЈ
!lstm_5/while/lstm_cell_3533/add_1AddV2#lstm_5/while/lstm_cell_3533/mul:z:0%lstm_5/while/lstm_cell_3533/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџd
%lstm_5/while/lstm_cell_3533/Sigmoid_2Sigmoid*lstm_5/while/lstm_cell_3533/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџd
"lstm_5/while/lstm_cell_3533/Relu_1Relu%lstm_5/while/lstm_cell_3533/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџdЗ
!lstm_5/while/lstm_cell_3533/mul_2Mul)lstm_5/while/lstm_cell_3533/Sigmoid_2:y:00lstm_5/while/lstm_cell_3533/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџdy
7lstm_5/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : 
1lstm_5/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_5_while_placeholder_1@lstm_5/while/TensorArrayV2Write/TensorListSetItem/index:output:0%lstm_5/while/lstm_cell_3533/mul_2:z:0*
_output_shapes
: *
element_dtype0:щшвT
lstm_5/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :q
lstm_5/while/addAddV2lstm_5_while_placeholderlstm_5/while/add/y:output:0*
T0*
_output_shapes
: V
lstm_5/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
lstm_5/while/add_1AddV2&lstm_5_while_lstm_5_while_loop_counterlstm_5/while/add_1/y:output:0*
T0*
_output_shapes
: n
lstm_5/while/IdentityIdentitylstm_5/while/add_1:z:0^lstm_5/while/NoOp*
T0*
_output_shapes
: 
lstm_5/while/Identity_1Identity,lstm_5_while_lstm_5_while_maximum_iterations^lstm_5/while/NoOp*
T0*
_output_shapes
: n
lstm_5/while/Identity_2Identitylstm_5/while/add:z:0^lstm_5/while/NoOp*
T0*
_output_shapes
: 
lstm_5/while/Identity_3IdentityAlstm_5/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_5/while/NoOp*
T0*
_output_shapes
: 
lstm_5/while/Identity_4Identity%lstm_5/while/lstm_cell_3533/mul_2:z:0^lstm_5/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџd
lstm_5/while/Identity_5Identity%lstm_5/while/lstm_cell_3533/add_1:z:0^lstm_5/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџdђ
lstm_5/while/NoOpNoOp3^lstm_5/while/lstm_cell_3533/BiasAdd/ReadVariableOp2^lstm_5/while/lstm_cell_3533/MatMul/ReadVariableOp4^lstm_5/while/lstm_cell_3533/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "7
lstm_5_while_identitylstm_5/while/Identity:output:0";
lstm_5_while_identity_1 lstm_5/while/Identity_1:output:0";
lstm_5_while_identity_2 lstm_5/while/Identity_2:output:0";
lstm_5_while_identity_3 lstm_5/while/Identity_3:output:0";
lstm_5_while_identity_4 lstm_5/while/Identity_4:output:0";
lstm_5_while_identity_5 lstm_5/while/Identity_5:output:0"L
#lstm_5_while_lstm_5_strided_slice_1%lstm_5_while_lstm_5_strided_slice_1_0"|
;lstm_5_while_lstm_cell_3533_biasadd_readvariableop_resource=lstm_5_while_lstm_cell_3533_biasadd_readvariableop_resource_0"~
<lstm_5_while_lstm_cell_3533_matmul_1_readvariableop_resource>lstm_5_while_lstm_cell_3533_matmul_1_readvariableop_resource_0"z
:lstm_5_while_lstm_cell_3533_matmul_readvariableop_resource<lstm_5_while_lstm_cell_3533_matmul_readvariableop_resource_0"Ф
_lstm_5_while_tensorarrayv2read_tensorlistgetitem_lstm_5_tensorarrayunstack_tensorlistfromtensoralstm_5_while_tensorarrayv2read_tensorlistgetitem_lstm_5_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :џџџџџџџџџd:џџџџџџџџџd: : : : : 2h
2lstm_5/while/lstm_cell_3533/BiasAdd/ReadVariableOp2lstm_5/while/lstm_cell_3533/BiasAdd/ReadVariableOp2f
1lstm_5/while/lstm_cell_3533/MatMul/ReadVariableOp1lstm_5/while/lstm_cell_3533/MatMul/ReadVariableOp2j
3lstm_5/while/lstm_cell_3533/MatMul_1/ReadVariableOp3lstm_5/while/lstm_cell_3533/MatMul_1/ReadVariableOp: 
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
й

L__inference_lstm_cell_3533_layer_call_and_return_conditional_losses_24306860

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
 
_user_specified_namestates"Е	L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Й
serving_defaultЅ
I
lstm_5_input9
serving_default_lstm_5_input:0џџџџџџџџџ<
dense_170
StatefulPartitionedCall:0џџџџџџџџџtensorflow/serving/predict:ЁЦ
Ї
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
signatures
#_self_saveable_object_factories"
_tf_keras_sequential
џ
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_random_generator
cell

state_spec
#_self_saveable_object_factories"
_tf_keras_rnn_layer
р
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
 bias
#!_self_saveable_object_factories"
_tf_keras_layer
р
"	variables
#trainable_variables
$regularization_losses
%	keras_api
&__call__
*'&call_and_return_all_conditional_losses

(kernel
)bias
#*_self_saveable_object_factories"
_tf_keras_layer
р
+	variables
,trainable_variables
-regularization_losses
.	keras_api
/__call__
*0&call_and_return_all_conditional_losses

1kernel
2bias
#3_self_saveable_object_factories"
_tf_keras_layer
_
40
51
62
3
 4
(5
)6
17
28"
trackable_list_wrapper
_
40
51
62
3
 4
(5
)6
17
28"
trackable_list_wrapper
 "
trackable_list_wrapper
Ъ
7non_trainable_variables

8layers
9metrics
:layer_regularization_losses
;layer_metrics
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
ё
<trace_0
=trace_1
>trace_2
?trace_32
/__inference_sequential_5_layer_call_fn_24307230
/__inference_sequential_5_layer_call_fn_24307628
/__inference_sequential_5_layer_call_fn_24307651
/__inference_sequential_5_layer_call_fn_24307522П
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
 z<trace_0z=trace_1z>trace_2z?trace_3
н
@trace_0
Atrace_1
Btrace_2
Ctrace_32ђ
J__inference_sequential_5_layer_call_and_return_conditional_losses_24307816
J__inference_sequential_5_layer_call_and_return_conditional_losses_24307981
J__inference_sequential_5_layer_call_and_return_conditional_losses_24307548
J__inference_sequential_5_layer_call_and_return_conditional_losses_24307574П
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
 z@trace_0zAtrace_1zBtrace_2zCtrace_3
гBа
#__inference__wrapped_model_24306645lstm_5_input"
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
Diter

Ebeta_1

Fbeta_2
	Gdecay
Hlearning_ratem m(m)m1m2m4m5m6mv v(v)v1v2v4v5v6v"
	optimizer
,
Iserving_default"
signature_map
 "
trackable_dict_wrapper
5
40
51
62"
trackable_list_wrapper
5
40
51
62"
trackable_list_wrapper
 "
trackable_list_wrapper
Й

Jstates
Knon_trainable_variables

Llayers
Mmetrics
Nlayer_regularization_losses
Olayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
ю
Ptrace_0
Qtrace_1
Rtrace_2
Strace_32
)__inference_lstm_5_layer_call_fn_24307992
)__inference_lstm_5_layer_call_fn_24308003
)__inference_lstm_5_layer_call_fn_24308014
)__inference_lstm_5_layer_call_fn_24308025д
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
 zPtrace_0zQtrace_1zRtrace_2zStrace_3
к
Ttrace_0
Utrace_1
Vtrace_2
Wtrace_32я
D__inference_lstm_5_layer_call_and_return_conditional_losses_24308170
D__inference_lstm_5_layer_call_and_return_conditional_losses_24308315
D__inference_lstm_5_layer_call_and_return_conditional_losses_24308460
D__inference_lstm_5_layer_call_and_return_conditional_losses_24308605д
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
 zTtrace_0zUtrace_1zVtrace_2zWtrace_3
C
#X_self_saveable_object_factories"
_generic_user_object

Y	variables
Ztrainable_variables
[regularization_losses
\	keras_api
]__call__
*^&call_and_return_all_conditional_losses
__random_generator
`
state_size

4kernel
5recurrent_kernel
6bias
#a_self_saveable_object_factories"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
0
 1"
trackable_list_wrapper
.
0
 1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
bnon_trainable_variables

clayers
dmetrics
elayer_regularization_losses
flayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
я
gtrace_02в
+__inference_dense_15_layer_call_fn_24308614Ђ
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

htrace_02э
F__inference_dense_15_layer_call_and_return_conditional_losses_24308625Ђ
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
!:d22dense_15/kernel
:22dense_15/bias
 "
trackable_dict_wrapper
.
(0
)1"
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
inon_trainable_variables

jlayers
kmetrics
llayer_regularization_losses
mlayer_metrics
"	variables
#trainable_variables
$regularization_losses
&__call__
*'&call_and_return_all_conditional_losses
&'"call_and_return_conditional_losses"
_generic_user_object
я
ntrace_02в
+__inference_dense_16_layer_call_fn_24308634Ђ
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

otrace_02э
F__inference_dense_16_layer_call_and_return_conditional_losses_24308645Ђ
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
!:222dense_16/kernel
:22dense_16/bias
 "
trackable_dict_wrapper
.
10
21"
trackable_list_wrapper
.
10
21"
trackable_list_wrapper
 "
trackable_list_wrapper
­
pnon_trainable_variables

qlayers
rmetrics
slayer_regularization_losses
tlayer_metrics
+	variables
,trainable_variables
-regularization_losses
/__call__
*0&call_and_return_all_conditional_losses
&0"call_and_return_conditional_losses"
_generic_user_object
я
utrace_02в
+__inference_dense_17_layer_call_fn_24308654Ђ
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
 zutrace_0

vtrace_02э
F__inference_dense_17_layer_call_and_return_conditional_losses_24308664Ђ
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
 zvtrace_0
!:22dense_17/kernel
:2dense_17/bias
 "
trackable_dict_wrapper
/:-	2lstm_5/lstm_cell_2959/kernel
9:7	d2&lstm_5/lstm_cell_2959/recurrent_kernel
):'2lstm_5/lstm_cell_2959/bias
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
.
w0
x1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
B
/__inference_sequential_5_layer_call_fn_24307230lstm_5_input"П
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
B§
/__inference_sequential_5_layer_call_fn_24307628inputs"П
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
B§
/__inference_sequential_5_layer_call_fn_24307651inputs"П
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
/__inference_sequential_5_layer_call_fn_24307522lstm_5_input"П
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
B
J__inference_sequential_5_layer_call_and_return_conditional_losses_24307816inputs"П
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
B
J__inference_sequential_5_layer_call_and_return_conditional_losses_24307981inputs"П
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
J__inference_sequential_5_layer_call_and_return_conditional_losses_24307548lstm_5_input"П
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
J__inference_sequential_5_layer_call_and_return_conditional_losses_24307574lstm_5_input"П
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
вBЯ
&__inference_signature_wrapper_24307605lstm_5_input"
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
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
B
)__inference_lstm_5_layer_call_fn_24307992inputs/0"д
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
B
)__inference_lstm_5_layer_call_fn_24308003inputs/0"д
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
B
)__inference_lstm_5_layer_call_fn_24308014inputs"д
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
B
)__inference_lstm_5_layer_call_fn_24308025inputs"д
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
ЌBЉ
D__inference_lstm_5_layer_call_and_return_conditional_losses_24308170inputs/0"д
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
ЌBЉ
D__inference_lstm_5_layer_call_and_return_conditional_losses_24308315inputs/0"д
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
ЊBЇ
D__inference_lstm_5_layer_call_and_return_conditional_losses_24308460inputs"д
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
ЊBЇ
D__inference_lstm_5_layer_call_and_return_conditional_losses_24308605inputs"д
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
 "
trackable_dict_wrapper
5
40
51
62"
trackable_list_wrapper
5
40
51
62"
trackable_list_wrapper
 "
trackable_list_wrapper
­
ynon_trainable_variables

zlayers
{metrics
|layer_regularization_losses
}layer_metrics
Y	variables
Ztrainable_variables
[regularization_losses
]__call__
*^&call_and_return_all_conditional_losses
&^"call_and_return_conditional_losses"
_generic_user_object
н
~trace_0
trace_12І
1__inference_lstm_cell_3533_layer_call_fn_24308681
1__inference_lstm_cell_3533_layer_call_fn_24308698Н
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
 z~trace_0ztrace_1

trace_0
trace_12м
L__inference_lstm_cell_3533_layer_call_and_return_conditional_losses_24308730
L__inference_lstm_cell_3533_layer_call_and_return_conditional_losses_24308762Н
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
 ztrace_0ztrace_1
D
$_self_saveable_object_factories"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
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
пBм
+__inference_dense_15_layer_call_fn_24308614inputs"Ђ
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
њBї
F__inference_dense_15_layer_call_and_return_conditional_losses_24308625inputs"Ђ
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
пBм
+__inference_dense_16_layer_call_fn_24308634inputs"Ђ
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
њBї
F__inference_dense_16_layer_call_and_return_conditional_losses_24308645inputs"Ђ
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
пBм
+__inference_dense_17_layer_call_fn_24308654inputs"Ђ
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
њBї
F__inference_dense_17_layer_call_and_return_conditional_losses_24308664inputs"Ђ
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
R
	variables
	keras_api

total

count"
_tf_keras_metric
c
	variables
	keras_api

total

count

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
B
1__inference_lstm_cell_3533_layer_call_fn_24308681inputsstates/0states/1"Н
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
B
1__inference_lstm_cell_3533_layer_call_fn_24308698inputsstates/0states/1"Н
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
ЏBЌ
L__inference_lstm_cell_3533_layer_call_and_return_conditional_losses_24308730inputsstates/0states/1"Н
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
ЏBЌ
L__inference_lstm_cell_3533_layer_call_and_return_conditional_losses_24308762inputsstates/0states/1"Н
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
 "
trackable_dict_wrapper
0
0
1"
trackable_list_wrapper
.
	variables"
_generic_user_object
:  (2total
:  (2count
0
0
1"
trackable_list_wrapper
.
	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
&:$d22Adam/dense_15/kernel/m
 :22Adam/dense_15/bias/m
&:$222Adam/dense_16/kernel/m
 :22Adam/dense_16/bias/m
&:$22Adam/dense_17/kernel/m
 :2Adam/dense_17/bias/m
4:2	2#Adam/lstm_5/lstm_cell_2959/kernel/m
>:<	d2-Adam/lstm_5/lstm_cell_2959/recurrent_kernel/m
.:,2!Adam/lstm_5/lstm_cell_2959/bias/m
&:$d22Adam/dense_15/kernel/v
 :22Adam/dense_15/bias/v
&:$222Adam/dense_16/kernel/v
 :22Adam/dense_16/bias/v
&:$22Adam/dense_17/kernel/v
 :2Adam/dense_17/bias/v
4:2	2#Adam/lstm_5/lstm_cell_2959/kernel/v
>:<	d2-Adam/lstm_5/lstm_cell_2959/recurrent_kernel/v
.:,2!Adam/lstm_5/lstm_cell_2959/bias/vЂ
#__inference__wrapped_model_24306645{	456 ()129Ђ6
/Ђ,
*'
lstm_5_inputџџџџџџџџџ
Њ "3Њ0
.
dense_17"
dense_17џџџџџџџџџІ
F__inference_dense_15_layer_call_and_return_conditional_losses_24308625\ /Ђ,
%Ђ"
 
inputsџџџџџџџџџd
Њ "%Ђ"

0џџџџџџџџџ2
 ~
+__inference_dense_15_layer_call_fn_24308614O /Ђ,
%Ђ"
 
inputsџџџџџџџџџd
Њ "џџџџџџџџџ2І
F__inference_dense_16_layer_call_and_return_conditional_losses_24308645\()/Ђ,
%Ђ"
 
inputsџџџџџџџџџ2
Њ "%Ђ"

0џџџџџџџџџ2
 ~
+__inference_dense_16_layer_call_fn_24308634O()/Ђ,
%Ђ"
 
inputsџџџџџџџџџ2
Њ "џџџџџџџџџ2І
F__inference_dense_17_layer_call_and_return_conditional_losses_24308664\12/Ђ,
%Ђ"
 
inputsџџџџџџџџџ2
Њ "%Ђ"

0џџџџџџџџџ
 ~
+__inference_dense_17_layer_call_fn_24308654O12/Ђ,
%Ђ"
 
inputsџџџџџџџџџ2
Њ "џџџџџџџџџХ
D__inference_lstm_5_layer_call_and_return_conditional_losses_24308170}456OЂL
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
 Х
D__inference_lstm_5_layer_call_and_return_conditional_losses_24308315}456OЂL
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
 Е
D__inference_lstm_5_layer_call_and_return_conditional_losses_24308460m456?Ђ<
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
 Е
D__inference_lstm_5_layer_call_and_return_conditional_losses_24308605m456?Ђ<
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
 
)__inference_lstm_5_layer_call_fn_24307992p456OЂL
EЂB
41
/,
inputs/0џџџџџџџџџџџџџџџџџџ

 
p 

 
Њ "џџџџџџџџџd
)__inference_lstm_5_layer_call_fn_24308003p456OЂL
EЂB
41
/,
inputs/0џџџџџџџџџџџџџџџџџџ

 
p

 
Њ "џџџџџџџџџd
)__inference_lstm_5_layer_call_fn_24308014`456?Ђ<
5Ђ2
$!
inputsџџџџџџџџџ

 
p 

 
Њ "џџџџџџџџџd
)__inference_lstm_5_layer_call_fn_24308025`456?Ђ<
5Ђ2
$!
inputsџџџџџџџџџ

 
p

 
Њ "џџџџџџџџџdЮ
L__inference_lstm_cell_3533_layer_call_and_return_conditional_losses_24308730§456Ђ}
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
 Ю
L__inference_lstm_cell_3533_layer_call_and_return_conditional_losses_24308762§456Ђ}
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
 Ѓ
1__inference_lstm_cell_3533_layer_call_fn_24308681э456Ђ}
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
1/1џџџџџџџџџdЃ
1__inference_lstm_cell_3533_layer_call_fn_24308698э456Ђ}
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
J__inference_sequential_5_layer_call_and_return_conditional_losses_24307548u	456 ()12AЂ>
7Ђ4
*'
lstm_5_inputџџџџџџџџџ
p 

 
Њ "%Ђ"

0џџџџџџџџџ
 У
J__inference_sequential_5_layer_call_and_return_conditional_losses_24307574u	456 ()12AЂ>
7Ђ4
*'
lstm_5_inputџџџџџџџџџ
p

 
Њ "%Ђ"

0џџџџџџџџџ
 Н
J__inference_sequential_5_layer_call_and_return_conditional_losses_24307816o	456 ()12;Ђ8
1Ђ.
$!
inputsџџџџџџџџџ
p 

 
Њ "%Ђ"

0џџџџџџџџџ
 Н
J__inference_sequential_5_layer_call_and_return_conditional_losses_24307981o	456 ()12;Ђ8
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
/__inference_sequential_5_layer_call_fn_24307230h	456 ()12AЂ>
7Ђ4
*'
lstm_5_inputџџџџџџџџџ
p 

 
Њ "џџџџџџџџџ
/__inference_sequential_5_layer_call_fn_24307522h	456 ()12AЂ>
7Ђ4
*'
lstm_5_inputџџџџџџџџџ
p

 
Њ "џџџџџџџџџ
/__inference_sequential_5_layer_call_fn_24307628b	456 ()12;Ђ8
1Ђ.
$!
inputsџџџџџџџџџ
p 

 
Њ "џџџџџџџџџ
/__inference_sequential_5_layer_call_fn_24307651b	456 ()12;Ђ8
1Ђ.
$!
inputsџџџџџџџџџ
p

 
Њ "џџџџџџџџџЖ
&__inference_signature_wrapper_24307605	456 ()12IЂF
Ђ 
?Њ<
:
lstm_5_input*'
lstm_5_inputџџџџџџџџџ"3Њ0
.
dense_17"
dense_17џџџџџџџџџ