ä
É
:
Add
x"T
y"T
z"T"
Ttype:
2	
î
	ApplyAdam
var"T	
m"T	
v"T
beta1_power"T
beta2_power"T
lr"T

beta1"T

beta2"T
epsilon"T	
grad"T
out"T" 
Ttype:
2	"
use_lockingbool( "
use_nesterovbool( 
x
Assign
ref"T

value"T

output_ref"T"	
Ttype"
validate_shapebool("
use_lockingbool(
R
BroadcastGradientArgs
s0"T
s1"T
r0"T
r1"T"
Ttype0:
2	
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
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
?
FloorDiv
x"T
y"T
z"T"
Ttype:
2	
.
Identity

input"T
output"T"	
Ttype
p
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
	2
;
Maximum
x"T
y"T
z"T"
Ttype:

2	

Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
=
Mul
x"T
y"T
z"T"
Ttype:
2	
.
Neg
x"T
y"T"
Ttype:

2	
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

Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	

RandomStandardNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
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
=
SigmoidGrad
y"T
dy"T
z"T"
Ttype:

2
1
Square
x"T
y"T"
Ttype:

2	
G
SquaredDifference
x"T
y"T
z"T"
Ttype:

2	
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
:
Sub
x"T
y"T
z"T"
Ttype:
2	

Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
s

VariableV2
ref"dtype"
shapeshape"
dtypetype"
	containerstring "
shared_namestring "serve*1.12.02b'v1.12.0-rc2-3-ga6d8ffae09'ď­
j
myInputPlaceholder*
shape:˙˙˙˙˙˙˙˙˙*
dtype0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
n
PlaceholderPlaceholder*
shape:˙˙˙˙˙˙˙˙˙*
dtype0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
d
random_normal/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
W
random_normal/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Y
random_normal/stddevConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 

"random_normal/RandomStandardNormalRandomStandardNormalrandom_normal/shape*

seedc*
T0*
dtype0*
_output_shapes
:	*
seed2
|
random_normal/mulMul"random_normal/RandomStandardNormalrandom_normal/stddev*
T0*
_output_shapes
:	
e
random_normalAddrandom_normal/mulrandom_normal/mean*
T0*
_output_shapes
:	
~
Variable
VariableV2*
shape:	*
shared_name *
dtype0*
_output_shapes
:	*
	container 
˘
Variable/AssignAssignVariablerandom_normal*
use_locking(*
T0*
_class
loc:@Variable*
validate_shape(*
_output_shapes
:	
j
Variable/readIdentityVariable*
T0*
_class
loc:@Variable*
_output_shapes
:	
`
random_normal_1/shapeConst*
valueB:*
dtype0*
_output_shapes
:
Y
random_normal_1/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
[
random_normal_1/stddevConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 

$random_normal_1/RandomStandardNormalRandomStandardNormalrandom_normal_1/shape*

seedc*
T0*
dtype0*
_output_shapes	
:*
seed2
~
random_normal_1/mulMul$random_normal_1/RandomStandardNormalrandom_normal_1/stddev*
T0*
_output_shapes	
:
g
random_normal_1Addrandom_normal_1/mulrandom_normal_1/mean*
T0*
_output_shapes	
:
x

Variable_1
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes	
:*
	container 
Ś
Variable_1/AssignAssign
Variable_1random_normal_1*
use_locking(*
T0*
_class
loc:@Variable_1*
validate_shape(*
_output_shapes	
:
l
Variable_1/readIdentity
Variable_1*
T0*
_class
loc:@Variable_1*
_output_shapes	
:
f
random_normal_2/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
Y
random_normal_2/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
[
random_normal_2/stddevConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
¤
$random_normal_2/RandomStandardNormalRandomStandardNormalrandom_normal_2/shape*

seedc*
T0*
dtype0* 
_output_shapes
:
*
seed2

random_normal_2/mulMul$random_normal_2/RandomStandardNormalrandom_normal_2/stddev*
T0* 
_output_shapes
:

l
random_normal_2Addrandom_normal_2/mulrandom_normal_2/mean*
T0* 
_output_shapes
:



Variable_2
VariableV2*
shape:
*
shared_name *
dtype0* 
_output_shapes
:
*
	container 
Ť
Variable_2/AssignAssign
Variable_2random_normal_2*
use_locking(*
T0*
_class
loc:@Variable_2*
validate_shape(* 
_output_shapes
:

q
Variable_2/readIdentity
Variable_2*
T0*
_class
loc:@Variable_2* 
_output_shapes
:

`
random_normal_3/shapeConst*
valueB:*
dtype0*
_output_shapes
:
Y
random_normal_3/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
[
random_normal_3/stddevConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 

$random_normal_3/RandomStandardNormalRandomStandardNormalrandom_normal_3/shape*

seedc*
T0*
dtype0*
_output_shapes	
:*
seed2 
~
random_normal_3/mulMul$random_normal_3/RandomStandardNormalrandom_normal_3/stddev*
T0*
_output_shapes	
:
g
random_normal_3Addrandom_normal_3/mulrandom_normal_3/mean*
T0*
_output_shapes	
:
x

Variable_3
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes	
:*
	container 
Ś
Variable_3/AssignAssign
Variable_3random_normal_3*
use_locking(*
T0*
_class
loc:@Variable_3*
validate_shape(*
_output_shapes	
:
l
Variable_3/readIdentity
Variable_3*
T0*
_class
loc:@Variable_3*
_output_shapes	
:
f
random_normal_4/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
Y
random_normal_4/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
[
random_normal_4/stddevConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
¤
$random_normal_4/RandomStandardNormalRandomStandardNormalrandom_normal_4/shape*

seedc*
T0*
dtype0* 
_output_shapes
:
*
seed2)

random_normal_4/mulMul$random_normal_4/RandomStandardNormalrandom_normal_4/stddev*
T0* 
_output_shapes
:

l
random_normal_4Addrandom_normal_4/mulrandom_normal_4/mean*
T0* 
_output_shapes
:



Variable_4
VariableV2*
shape:
*
shared_name *
dtype0* 
_output_shapes
:
*
	container 
Ť
Variable_4/AssignAssign
Variable_4random_normal_4*
use_locking(*
T0*
_class
loc:@Variable_4*
validate_shape(* 
_output_shapes
:

q
Variable_4/readIdentity
Variable_4*
T0*
_class
loc:@Variable_4* 
_output_shapes
:

`
random_normal_5/shapeConst*
valueB:*
dtype0*
_output_shapes
:
Y
random_normal_5/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
[
random_normal_5/stddevConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 

$random_normal_5/RandomStandardNormalRandomStandardNormalrandom_normal_5/shape*

seedc*
T0*
dtype0*
_output_shapes	
:*
seed22
~
random_normal_5/mulMul$random_normal_5/RandomStandardNormalrandom_normal_5/stddev*
T0*
_output_shapes	
:
g
random_normal_5Addrandom_normal_5/mulrandom_normal_5/mean*
T0*
_output_shapes	
:
x

Variable_5
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes	
:*
	container 
Ś
Variable_5/AssignAssign
Variable_5random_normal_5*
use_locking(*
T0*
_class
loc:@Variable_5*
validate_shape(*
_output_shapes	
:
l
Variable_5/readIdentity
Variable_5*
T0*
_class
loc:@Variable_5*
_output_shapes	
:
f
random_normal_6/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
Y
random_normal_6/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
[
random_normal_6/stddevConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
¤
$random_normal_6/RandomStandardNormalRandomStandardNormalrandom_normal_6/shape*

seedc*
T0*
dtype0* 
_output_shapes
:
*
seed2;

random_normal_6/mulMul$random_normal_6/RandomStandardNormalrandom_normal_6/stddev*
T0* 
_output_shapes
:

l
random_normal_6Addrandom_normal_6/mulrandom_normal_6/mean*
T0* 
_output_shapes
:



Variable_6
VariableV2*
shape:
*
shared_name *
dtype0* 
_output_shapes
:
*
	container 
Ť
Variable_6/AssignAssign
Variable_6random_normal_6*
use_locking(*
T0*
_class
loc:@Variable_6*
validate_shape(* 
_output_shapes
:

q
Variable_6/readIdentity
Variable_6*
T0*
_class
loc:@Variable_6* 
_output_shapes
:

`
random_normal_7/shapeConst*
valueB:*
dtype0*
_output_shapes
:
Y
random_normal_7/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
[
random_normal_7/stddevConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 

$random_normal_7/RandomStandardNormalRandomStandardNormalrandom_normal_7/shape*

seedc*
T0*
dtype0*
_output_shapes	
:*
seed2D
~
random_normal_7/mulMul$random_normal_7/RandomStandardNormalrandom_normal_7/stddev*
T0*
_output_shapes	
:
g
random_normal_7Addrandom_normal_7/mulrandom_normal_7/mean*
T0*
_output_shapes	
:
x

Variable_7
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes	
:*
	container 
Ś
Variable_7/AssignAssign
Variable_7random_normal_7*
use_locking(*
T0*
_class
loc:@Variable_7*
validate_shape(*
_output_shapes	
:
l
Variable_7/readIdentity
Variable_7*
T0*
_class
loc:@Variable_7*
_output_shapes	
:
f
random_normal_8/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
Y
random_normal_8/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
[
random_normal_8/stddevConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
¤
$random_normal_8/RandomStandardNormalRandomStandardNormalrandom_normal_8/shape*

seedc*
T0*
dtype0* 
_output_shapes
:
*
seed2M

random_normal_8/mulMul$random_normal_8/RandomStandardNormalrandom_normal_8/stddev*
T0* 
_output_shapes
:

l
random_normal_8Addrandom_normal_8/mulrandom_normal_8/mean*
T0* 
_output_shapes
:



Variable_8
VariableV2*
shape:
*
shared_name *
dtype0* 
_output_shapes
:
*
	container 
Ť
Variable_8/AssignAssign
Variable_8random_normal_8*
use_locking(*
T0*
_class
loc:@Variable_8*
validate_shape(* 
_output_shapes
:

q
Variable_8/readIdentity
Variable_8*
T0*
_class
loc:@Variable_8* 
_output_shapes
:

`
random_normal_9/shapeConst*
valueB:*
dtype0*
_output_shapes
:
Y
random_normal_9/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
[
random_normal_9/stddevConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 

$random_normal_9/RandomStandardNormalRandomStandardNormalrandom_normal_9/shape*

seedc*
T0*
dtype0*
_output_shapes	
:*
seed2V
~
random_normal_9/mulMul$random_normal_9/RandomStandardNormalrandom_normal_9/stddev*
T0*
_output_shapes	
:
g
random_normal_9Addrandom_normal_9/mulrandom_normal_9/mean*
T0*
_output_shapes	
:
x

Variable_9
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes	
:*
	container 
Ś
Variable_9/AssignAssign
Variable_9random_normal_9*
use_locking(*
T0*
_class
loc:@Variable_9*
validate_shape(*
_output_shapes	
:
l
Variable_9/readIdentity
Variable_9*
T0*
_class
loc:@Variable_9*
_output_shapes	
:
g
random_normal_10/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
Z
random_normal_10/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
\
random_normal_10/stddevConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
Ś
%random_normal_10/RandomStandardNormalRandomStandardNormalrandom_normal_10/shape*

seedc*
T0*
dtype0* 
_output_shapes
:
*
seed2_

random_normal_10/mulMul%random_normal_10/RandomStandardNormalrandom_normal_10/stddev*
T0* 
_output_shapes
:

o
random_normal_10Addrandom_normal_10/mulrandom_normal_10/mean*
T0* 
_output_shapes
:


Variable_10
VariableV2*
shape:
*
shared_name *
dtype0* 
_output_shapes
:
*
	container 
Ż
Variable_10/AssignAssignVariable_10random_normal_10*
use_locking(*
T0*
_class
loc:@Variable_10*
validate_shape(* 
_output_shapes
:

t
Variable_10/readIdentityVariable_10*
T0*
_class
loc:@Variable_10* 
_output_shapes
:

a
random_normal_11/shapeConst*
valueB:*
dtype0*
_output_shapes
:
Z
random_normal_11/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
\
random_normal_11/stddevConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
Ą
%random_normal_11/RandomStandardNormalRandomStandardNormalrandom_normal_11/shape*

seedc*
T0*
dtype0*
_output_shapes	
:*
seed2h

random_normal_11/mulMul%random_normal_11/RandomStandardNormalrandom_normal_11/stddev*
T0*
_output_shapes	
:
j
random_normal_11Addrandom_normal_11/mulrandom_normal_11/mean*
T0*
_output_shapes	
:
y
Variable_11
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes	
:*
	container 
Ş
Variable_11/AssignAssignVariable_11random_normal_11*
use_locking(*
T0*
_class
loc:@Variable_11*
validate_shape(*
_output_shapes	
:
o
Variable_11/readIdentityVariable_11*
T0*
_class
loc:@Variable_11*
_output_shapes	
:
g
random_normal_12/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
Z
random_normal_12/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
\
random_normal_12/stddevConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
Ś
%random_normal_12/RandomStandardNormalRandomStandardNormalrandom_normal_12/shape*

seedc*
T0*
dtype0* 
_output_shapes
:
*
seed2q

random_normal_12/mulMul%random_normal_12/RandomStandardNormalrandom_normal_12/stddev*
T0* 
_output_shapes
:

o
random_normal_12Addrandom_normal_12/mulrandom_normal_12/mean*
T0* 
_output_shapes
:


Variable_12
VariableV2*
shape:
*
shared_name *
dtype0* 
_output_shapes
:
*
	container 
Ż
Variable_12/AssignAssignVariable_12random_normal_12*
use_locking(*
T0*
_class
loc:@Variable_12*
validate_shape(* 
_output_shapes
:

t
Variable_12/readIdentityVariable_12*
T0*
_class
loc:@Variable_12* 
_output_shapes
:

a
random_normal_13/shapeConst*
valueB:*
dtype0*
_output_shapes
:
Z
random_normal_13/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
\
random_normal_13/stddevConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
Ą
%random_normal_13/RandomStandardNormalRandomStandardNormalrandom_normal_13/shape*

seedc*
T0*
dtype0*
_output_shapes	
:*
seed2z

random_normal_13/mulMul%random_normal_13/RandomStandardNormalrandom_normal_13/stddev*
T0*
_output_shapes	
:
j
random_normal_13Addrandom_normal_13/mulrandom_normal_13/mean*
T0*
_output_shapes	
:
y
Variable_13
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes	
:*
	container 
Ş
Variable_13/AssignAssignVariable_13random_normal_13*
use_locking(*
T0*
_class
loc:@Variable_13*
validate_shape(*
_output_shapes	
:
o
Variable_13/readIdentityVariable_13*
T0*
_class
loc:@Variable_13*
_output_shapes	
:
g
random_normal_14/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
Z
random_normal_14/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
\
random_normal_14/stddevConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
§
%random_normal_14/RandomStandardNormalRandomStandardNormalrandom_normal_14/shape*

seedc*
T0*
dtype0* 
_output_shapes
:
*
seed2

random_normal_14/mulMul%random_normal_14/RandomStandardNormalrandom_normal_14/stddev*
T0* 
_output_shapes
:

o
random_normal_14Addrandom_normal_14/mulrandom_normal_14/mean*
T0* 
_output_shapes
:


Variable_14
VariableV2*
shape:
*
shared_name *
dtype0* 
_output_shapes
:
*
	container 
Ż
Variable_14/AssignAssignVariable_14random_normal_14*
use_locking(*
T0*
_class
loc:@Variable_14*
validate_shape(* 
_output_shapes
:

t
Variable_14/readIdentityVariable_14*
T0*
_class
loc:@Variable_14* 
_output_shapes
:

a
random_normal_15/shapeConst*
valueB:*
dtype0*
_output_shapes
:
Z
random_normal_15/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
\
random_normal_15/stddevConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
˘
%random_normal_15/RandomStandardNormalRandomStandardNormalrandom_normal_15/shape*

seedc*
T0*
dtype0*
_output_shapes	
:*
seed2

random_normal_15/mulMul%random_normal_15/RandomStandardNormalrandom_normal_15/stddev*
T0*
_output_shapes	
:
j
random_normal_15Addrandom_normal_15/mulrandom_normal_15/mean*
T0*
_output_shapes	
:
y
Variable_15
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes	
:*
	container 
Ş
Variable_15/AssignAssignVariable_15random_normal_15*
use_locking(*
T0*
_class
loc:@Variable_15*
validate_shape(*
_output_shapes	
:
o
Variable_15/readIdentityVariable_15*
T0*
_class
loc:@Variable_15*
_output_shapes	
:
g
random_normal_16/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
Z
random_normal_16/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
\
random_normal_16/stddevConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
Ś
%random_normal_16/RandomStandardNormalRandomStandardNormalrandom_normal_16/shape*

seedc*
T0*
dtype0*
_output_shapes
:	*
seed2

random_normal_16/mulMul%random_normal_16/RandomStandardNormalrandom_normal_16/stddev*
T0*
_output_shapes
:	
n
random_normal_16Addrandom_normal_16/mulrandom_normal_16/mean*
T0*
_output_shapes
:	

Variable_16
VariableV2*
shape:	*
shared_name *
dtype0*
_output_shapes
:	*
	container 
Ž
Variable_16/AssignAssignVariable_16random_normal_16*
use_locking(*
T0*
_class
loc:@Variable_16*
validate_shape(*
_output_shapes
:	
s
Variable_16/readIdentityVariable_16*
T0*
_class
loc:@Variable_16*
_output_shapes
:	
`
random_normal_17/shapeConst*
valueB:*
dtype0*
_output_shapes
:
Z
random_normal_17/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
\
random_normal_17/stddevConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
Ą
%random_normal_17/RandomStandardNormalRandomStandardNormalrandom_normal_17/shape*

seedc*
T0*
dtype0*
_output_shapes
:*
seed2

random_normal_17/mulMul%random_normal_17/RandomStandardNormalrandom_normal_17/stddev*
T0*
_output_shapes
:
i
random_normal_17Addrandom_normal_17/mulrandom_normal_17/mean*
T0*
_output_shapes
:
w
Variable_17
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes
:*
	container 
Š
Variable_17/AssignAssignVariable_17random_normal_17*
use_locking(*
T0*
_class
loc:@Variable_17*
validate_shape(*
_output_shapes
:
n
Variable_17/readIdentityVariable_17*
T0*
_class
loc:@Variable_17*
_output_shapes
:
g
random_normal_18/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
Z
random_normal_18/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
\
random_normal_18/stddevConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
Ľ
%random_normal_18/RandomStandardNormalRandomStandardNormalrandom_normal_18/shape*

seedc*
T0*
dtype0*
_output_shapes

:*
seed2§

random_normal_18/mulMul%random_normal_18/RandomStandardNormalrandom_normal_18/stddev*
T0*
_output_shapes

:
m
random_normal_18Addrandom_normal_18/mulrandom_normal_18/mean*
T0*
_output_shapes

:

Variable_18
VariableV2*
shape
:*
shared_name *
dtype0*
_output_shapes

:*
	container 
­
Variable_18/AssignAssignVariable_18random_normal_18*
use_locking(*
T0*
_class
loc:@Variable_18*
validate_shape(*
_output_shapes

:
r
Variable_18/readIdentityVariable_18*
T0*
_class
loc:@Variable_18*
_output_shapes

:
`
random_normal_19/shapeConst*
valueB:*
dtype0*
_output_shapes
:
Z
random_normal_19/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
\
random_normal_19/stddevConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
Ą
%random_normal_19/RandomStandardNormalRandomStandardNormalrandom_normal_19/shape*

seedc*
T0*
dtype0*
_output_shapes
:*
seed2°

random_normal_19/mulMul%random_normal_19/RandomStandardNormalrandom_normal_19/stddev*
T0*
_output_shapes
:
i
random_normal_19Addrandom_normal_19/mulrandom_normal_19/mean*
T0*
_output_shapes
:
w
Variable_19
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes
:*
	container 
Š
Variable_19/AssignAssignVariable_19random_normal_19*
use_locking(*
T0*
_class
loc:@Variable_19*
validate_shape(*
_output_shapes
:
n
Variable_19/readIdentityVariable_19*
T0*
_class
loc:@Variable_19*
_output_shapes
:

MatMulMatMulmyInputVariable/read*
transpose_b( *
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( 
V
addAddMatMulVariable_1/read*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
J
SigmoidSigmoidadd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

MatMul_1MatMulSigmoidVariable_2/read*
transpose_b( *
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( 
Z
add_1AddMatMul_1Variable_3/read*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
N
	Sigmoid_1Sigmoidadd_1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

MatMul_2MatMul	Sigmoid_1Variable_4/read*
transpose_b( *
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( 
Z
add_2AddMatMul_2Variable_5/read*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
N
	Sigmoid_2Sigmoidadd_2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

MatMul_3MatMul	Sigmoid_2Variable_6/read*
transpose_b( *
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( 
Z
add_3AddMatMul_3Variable_7/read*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
N
	Sigmoid_3Sigmoidadd_3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

MatMul_4MatMul	Sigmoid_3Variable_8/read*
transpose_b( *
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( 
Z
add_4AddMatMul_4Variable_9/read*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
N
	Sigmoid_4Sigmoidadd_4*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

MatMul_5MatMul	Sigmoid_4Variable_10/read*
transpose_b( *
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( 
[
add_5AddMatMul_5Variable_11/read*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
N
	Sigmoid_5Sigmoidadd_5*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

MatMul_6MatMul	Sigmoid_5Variable_12/read*
transpose_b( *
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( 
[
add_6AddMatMul_6Variable_13/read*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
N
	Sigmoid_6Sigmoidadd_6*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

MatMul_7MatMul	Sigmoid_6Variable_14/read*
transpose_b( *
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( 
[
add_7AddMatMul_7Variable_15/read*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
N
	Sigmoid_7Sigmoidadd_7*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

MatMul_8MatMul	Sigmoid_7Variable_16/read*
transpose_b( *
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( 
Z
add_8AddMatMul_8Variable_17/read*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
M
	Sigmoid_8Sigmoidadd_8*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

myOutputMatMul	Sigmoid_8Variable_18/read*
transpose_b( *
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( 
Z
add_9AddmyOutputVariable_19/read*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
l
SquaredDifferenceSquaredDifferenceadd_9Placeholder*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
V
ConstConst*
valueB"       *
dtype0*
_output_shapes
:
d
MeanMeanSquaredDifferenceConst*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
R
gradients/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
X
gradients/grad_ys_0Const*
valueB
 *  ?*
dtype0*
_output_shapes
: 
o
gradients/FillFillgradients/Shapegradients/grad_ys_0*
T0*

index_type0*
_output_shapes
: 
r
!gradients/Mean_grad/Reshape/shapeConst*
valueB"      *
dtype0*
_output_shapes
:

gradients/Mean_grad/ReshapeReshapegradients/Fill!gradients/Mean_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes

:
j
gradients/Mean_grad/ShapeShapeSquaredDifference*
T0*
out_type0*
_output_shapes
:

gradients/Mean_grad/TileTilegradients/Mean_grad/Reshapegradients/Mean_grad/Shape*

Tmultiples0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
l
gradients/Mean_grad/Shape_1ShapeSquaredDifference*
T0*
out_type0*
_output_shapes
:
^
gradients/Mean_grad/Shape_2Const*
valueB *
dtype0*
_output_shapes
: 
c
gradients/Mean_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:

gradients/Mean_grad/ProdProdgradients/Mean_grad/Shape_1gradients/Mean_grad/Const*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
e
gradients/Mean_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:

gradients/Mean_grad/Prod_1Prodgradients/Mean_grad/Shape_2gradients/Mean_grad/Const_1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
_
gradients/Mean_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 

gradients/Mean_grad/MaximumMaximumgradients/Mean_grad/Prod_1gradients/Mean_grad/Maximum/y*
T0*
_output_shapes
: 

gradients/Mean_grad/floordivFloorDivgradients/Mean_grad/Prodgradients/Mean_grad/Maximum*
T0*
_output_shapes
: 
~
gradients/Mean_grad/CastCastgradients/Mean_grad/floordiv*

SrcT0*
Truncate( *
_output_shapes
: *

DstT0

gradients/Mean_grad/truedivRealDivgradients/Mean_grad/Tilegradients/Mean_grad/Cast*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
k
&gradients/SquaredDifference_grad/ShapeShapeadd_9*
T0*
out_type0*
_output_shapes
:
s
(gradients/SquaredDifference_grad/Shape_1ShapePlaceholder*
T0*
out_type0*
_output_shapes
:
Ţ
6gradients/SquaredDifference_grad/BroadcastGradientArgsBroadcastGradientArgs&gradients/SquaredDifference_grad/Shape(gradients/SquaredDifference_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙

'gradients/SquaredDifference_grad/scalarConst^gradients/Mean_grad/truediv*
valueB
 *   @*
dtype0*
_output_shapes
: 
Ł
$gradients/SquaredDifference_grad/mulMul'gradients/SquaredDifference_grad/scalargradients/Mean_grad/truediv*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

$gradients/SquaredDifference_grad/subSubadd_9Placeholder^gradients/Mean_grad/truediv*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ť
&gradients/SquaredDifference_grad/mul_1Mul$gradients/SquaredDifference_grad/mul$gradients/SquaredDifference_grad/sub*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ë
$gradients/SquaredDifference_grad/SumSum&gradients/SquaredDifference_grad/mul_16gradients/SquaredDifference_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
Á
(gradients/SquaredDifference_grad/ReshapeReshape$gradients/SquaredDifference_grad/Sum&gradients/SquaredDifference_grad/Shape*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ď
&gradients/SquaredDifference_grad/Sum_1Sum&gradients/SquaredDifference_grad/mul_18gradients/SquaredDifference_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
Ç
*gradients/SquaredDifference_grad/Reshape_1Reshape&gradients/SquaredDifference_grad/Sum_1(gradients/SquaredDifference_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

$gradients/SquaredDifference_grad/NegNeg*gradients/SquaredDifference_grad/Reshape_1*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

1gradients/SquaredDifference_grad/tuple/group_depsNoOp%^gradients/SquaredDifference_grad/Neg)^gradients/SquaredDifference_grad/Reshape

9gradients/SquaredDifference_grad/tuple/control_dependencyIdentity(gradients/SquaredDifference_grad/Reshape2^gradients/SquaredDifference_grad/tuple/group_deps*
T0*;
_class1
/-loc:@gradients/SquaredDifference_grad/Reshape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

;gradients/SquaredDifference_grad/tuple/control_dependency_1Identity$gradients/SquaredDifference_grad/Neg2^gradients/SquaredDifference_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients/SquaredDifference_grad/Neg*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
b
gradients/add_9_grad/ShapeShapemyOutput*
T0*
out_type0*
_output_shapes
:
f
gradients/add_9_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
ş
*gradients/add_9_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_9_grad/Shapegradients/add_9_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
Ć
gradients/add_9_grad/SumSum9gradients/SquaredDifference_grad/tuple/control_dependency*gradients/add_9_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:

gradients/add_9_grad/ReshapeReshapegradients/add_9_grad/Sumgradients/add_9_grad/Shape*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ę
gradients/add_9_grad/Sum_1Sum9gradients/SquaredDifference_grad/tuple/control_dependency,gradients/add_9_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:

gradients/add_9_grad/Reshape_1Reshapegradients/add_9_grad/Sum_1gradients/add_9_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
m
%gradients/add_9_grad/tuple/group_depsNoOp^gradients/add_9_grad/Reshape^gradients/add_9_grad/Reshape_1
â
-gradients/add_9_grad/tuple/control_dependencyIdentitygradients/add_9_grad/Reshape&^gradients/add_9_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/add_9_grad/Reshape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ű
/gradients/add_9_grad/tuple/control_dependency_1Identitygradients/add_9_grad/Reshape_1&^gradients/add_9_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/add_9_grad/Reshape_1*
_output_shapes
:
Á
gradients/myOutput_grad/MatMulMatMul-gradients/add_9_grad/tuple/control_dependencyVariable_18/read*
transpose_b(*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( 
ł
 gradients/myOutput_grad/MatMul_1MatMul	Sigmoid_8-gradients/add_9_grad/tuple/control_dependency*
transpose_b( *
T0*
_output_shapes

:*
transpose_a(
t
(gradients/myOutput_grad/tuple/group_depsNoOp^gradients/myOutput_grad/MatMul!^gradients/myOutput_grad/MatMul_1
ě
0gradients/myOutput_grad/tuple/control_dependencyIdentitygradients/myOutput_grad/MatMul)^gradients/myOutput_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/myOutput_grad/MatMul*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
é
2gradients/myOutput_grad/tuple/control_dependency_1Identity gradients/myOutput_grad/MatMul_1)^gradients/myOutput_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients/myOutput_grad/MatMul_1*
_output_shapes

:
˘
$gradients/Sigmoid_8_grad/SigmoidGradSigmoidGrad	Sigmoid_80gradients/myOutput_grad/tuple/control_dependency*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
b
gradients/add_8_grad/ShapeShapeMatMul_8*
T0*
out_type0*
_output_shapes
:
f
gradients/add_8_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
ş
*gradients/add_8_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_8_grad/Shapegradients/add_8_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
ą
gradients/add_8_grad/SumSum$gradients/Sigmoid_8_grad/SigmoidGrad*gradients/add_8_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:

gradients/add_8_grad/ReshapeReshapegradients/add_8_grad/Sumgradients/add_8_grad/Shape*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ľ
gradients/add_8_grad/Sum_1Sum$gradients/Sigmoid_8_grad/SigmoidGrad,gradients/add_8_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:

gradients/add_8_grad/Reshape_1Reshapegradients/add_8_grad/Sum_1gradients/add_8_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
m
%gradients/add_8_grad/tuple/group_depsNoOp^gradients/add_8_grad/Reshape^gradients/add_8_grad/Reshape_1
â
-gradients/add_8_grad/tuple/control_dependencyIdentitygradients/add_8_grad/Reshape&^gradients/add_8_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/add_8_grad/Reshape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ű
/gradients/add_8_grad/tuple/control_dependency_1Identitygradients/add_8_grad/Reshape_1&^gradients/add_8_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/add_8_grad/Reshape_1*
_output_shapes
:
Â
gradients/MatMul_8_grad/MatMulMatMul-gradients/add_8_grad/tuple/control_dependencyVariable_16/read*
transpose_b(*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( 
´
 gradients/MatMul_8_grad/MatMul_1MatMul	Sigmoid_7-gradients/add_8_grad/tuple/control_dependency*
transpose_b( *
T0*
_output_shapes
:	*
transpose_a(
t
(gradients/MatMul_8_grad/tuple/group_depsNoOp^gradients/MatMul_8_grad/MatMul!^gradients/MatMul_8_grad/MatMul_1
í
0gradients/MatMul_8_grad/tuple/control_dependencyIdentitygradients/MatMul_8_grad/MatMul)^gradients/MatMul_8_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/MatMul_8_grad/MatMul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ę
2gradients/MatMul_8_grad/tuple/control_dependency_1Identity gradients/MatMul_8_grad/MatMul_1)^gradients/MatMul_8_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients/MatMul_8_grad/MatMul_1*
_output_shapes
:	
Ł
$gradients/Sigmoid_7_grad/SigmoidGradSigmoidGrad	Sigmoid_70gradients/MatMul_8_grad/tuple/control_dependency*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
b
gradients/add_7_grad/ShapeShapeMatMul_7*
T0*
out_type0*
_output_shapes
:
g
gradients/add_7_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
ş
*gradients/add_7_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_7_grad/Shapegradients/add_7_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
ą
gradients/add_7_grad/SumSum$gradients/Sigmoid_7_grad/SigmoidGrad*gradients/add_7_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:

gradients/add_7_grad/ReshapeReshapegradients/add_7_grad/Sumgradients/add_7_grad/Shape*
T0*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ľ
gradients/add_7_grad/Sum_1Sum$gradients/Sigmoid_7_grad/SigmoidGrad,gradients/add_7_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:

gradients/add_7_grad/Reshape_1Reshapegradients/add_7_grad/Sum_1gradients/add_7_grad/Shape_1*
T0*
Tshape0*
_output_shapes	
:
m
%gradients/add_7_grad/tuple/group_depsNoOp^gradients/add_7_grad/Reshape^gradients/add_7_grad/Reshape_1
ă
-gradients/add_7_grad/tuple/control_dependencyIdentitygradients/add_7_grad/Reshape&^gradients/add_7_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/add_7_grad/Reshape*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ü
/gradients/add_7_grad/tuple/control_dependency_1Identitygradients/add_7_grad/Reshape_1&^gradients/add_7_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/add_7_grad/Reshape_1*
_output_shapes	
:
Â
gradients/MatMul_7_grad/MatMulMatMul-gradients/add_7_grad/tuple/control_dependencyVariable_14/read*
transpose_b(*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( 
ľ
 gradients/MatMul_7_grad/MatMul_1MatMul	Sigmoid_6-gradients/add_7_grad/tuple/control_dependency*
transpose_b( *
T0* 
_output_shapes
:
*
transpose_a(
t
(gradients/MatMul_7_grad/tuple/group_depsNoOp^gradients/MatMul_7_grad/MatMul!^gradients/MatMul_7_grad/MatMul_1
í
0gradients/MatMul_7_grad/tuple/control_dependencyIdentitygradients/MatMul_7_grad/MatMul)^gradients/MatMul_7_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/MatMul_7_grad/MatMul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ë
2gradients/MatMul_7_grad/tuple/control_dependency_1Identity gradients/MatMul_7_grad/MatMul_1)^gradients/MatMul_7_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients/MatMul_7_grad/MatMul_1* 
_output_shapes
:

Ł
$gradients/Sigmoid_6_grad/SigmoidGradSigmoidGrad	Sigmoid_60gradients/MatMul_7_grad/tuple/control_dependency*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
b
gradients/add_6_grad/ShapeShapeMatMul_6*
T0*
out_type0*
_output_shapes
:
g
gradients/add_6_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
ş
*gradients/add_6_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_6_grad/Shapegradients/add_6_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
ą
gradients/add_6_grad/SumSum$gradients/Sigmoid_6_grad/SigmoidGrad*gradients/add_6_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:

gradients/add_6_grad/ReshapeReshapegradients/add_6_grad/Sumgradients/add_6_grad/Shape*
T0*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ľ
gradients/add_6_grad/Sum_1Sum$gradients/Sigmoid_6_grad/SigmoidGrad,gradients/add_6_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:

gradients/add_6_grad/Reshape_1Reshapegradients/add_6_grad/Sum_1gradients/add_6_grad/Shape_1*
T0*
Tshape0*
_output_shapes	
:
m
%gradients/add_6_grad/tuple/group_depsNoOp^gradients/add_6_grad/Reshape^gradients/add_6_grad/Reshape_1
ă
-gradients/add_6_grad/tuple/control_dependencyIdentitygradients/add_6_grad/Reshape&^gradients/add_6_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/add_6_grad/Reshape*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ü
/gradients/add_6_grad/tuple/control_dependency_1Identitygradients/add_6_grad/Reshape_1&^gradients/add_6_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/add_6_grad/Reshape_1*
_output_shapes	
:
Â
gradients/MatMul_6_grad/MatMulMatMul-gradients/add_6_grad/tuple/control_dependencyVariable_12/read*
transpose_b(*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( 
ľ
 gradients/MatMul_6_grad/MatMul_1MatMul	Sigmoid_5-gradients/add_6_grad/tuple/control_dependency*
transpose_b( *
T0* 
_output_shapes
:
*
transpose_a(
t
(gradients/MatMul_6_grad/tuple/group_depsNoOp^gradients/MatMul_6_grad/MatMul!^gradients/MatMul_6_grad/MatMul_1
í
0gradients/MatMul_6_grad/tuple/control_dependencyIdentitygradients/MatMul_6_grad/MatMul)^gradients/MatMul_6_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/MatMul_6_grad/MatMul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ë
2gradients/MatMul_6_grad/tuple/control_dependency_1Identity gradients/MatMul_6_grad/MatMul_1)^gradients/MatMul_6_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients/MatMul_6_grad/MatMul_1* 
_output_shapes
:

Ł
$gradients/Sigmoid_5_grad/SigmoidGradSigmoidGrad	Sigmoid_50gradients/MatMul_6_grad/tuple/control_dependency*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
b
gradients/add_5_grad/ShapeShapeMatMul_5*
T0*
out_type0*
_output_shapes
:
g
gradients/add_5_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
ş
*gradients/add_5_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_5_grad/Shapegradients/add_5_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
ą
gradients/add_5_grad/SumSum$gradients/Sigmoid_5_grad/SigmoidGrad*gradients/add_5_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:

gradients/add_5_grad/ReshapeReshapegradients/add_5_grad/Sumgradients/add_5_grad/Shape*
T0*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ľ
gradients/add_5_grad/Sum_1Sum$gradients/Sigmoid_5_grad/SigmoidGrad,gradients/add_5_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:

gradients/add_5_grad/Reshape_1Reshapegradients/add_5_grad/Sum_1gradients/add_5_grad/Shape_1*
T0*
Tshape0*
_output_shapes	
:
m
%gradients/add_5_grad/tuple/group_depsNoOp^gradients/add_5_grad/Reshape^gradients/add_5_grad/Reshape_1
ă
-gradients/add_5_grad/tuple/control_dependencyIdentitygradients/add_5_grad/Reshape&^gradients/add_5_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/add_5_grad/Reshape*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ü
/gradients/add_5_grad/tuple/control_dependency_1Identitygradients/add_5_grad/Reshape_1&^gradients/add_5_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/add_5_grad/Reshape_1*
_output_shapes	
:
Â
gradients/MatMul_5_grad/MatMulMatMul-gradients/add_5_grad/tuple/control_dependencyVariable_10/read*
transpose_b(*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( 
ľ
 gradients/MatMul_5_grad/MatMul_1MatMul	Sigmoid_4-gradients/add_5_grad/tuple/control_dependency*
transpose_b( *
T0* 
_output_shapes
:
*
transpose_a(
t
(gradients/MatMul_5_grad/tuple/group_depsNoOp^gradients/MatMul_5_grad/MatMul!^gradients/MatMul_5_grad/MatMul_1
í
0gradients/MatMul_5_grad/tuple/control_dependencyIdentitygradients/MatMul_5_grad/MatMul)^gradients/MatMul_5_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/MatMul_5_grad/MatMul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ë
2gradients/MatMul_5_grad/tuple/control_dependency_1Identity gradients/MatMul_5_grad/MatMul_1)^gradients/MatMul_5_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients/MatMul_5_grad/MatMul_1* 
_output_shapes
:

Ł
$gradients/Sigmoid_4_grad/SigmoidGradSigmoidGrad	Sigmoid_40gradients/MatMul_5_grad/tuple/control_dependency*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
b
gradients/add_4_grad/ShapeShapeMatMul_4*
T0*
out_type0*
_output_shapes
:
g
gradients/add_4_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
ş
*gradients/add_4_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_4_grad/Shapegradients/add_4_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
ą
gradients/add_4_grad/SumSum$gradients/Sigmoid_4_grad/SigmoidGrad*gradients/add_4_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:

gradients/add_4_grad/ReshapeReshapegradients/add_4_grad/Sumgradients/add_4_grad/Shape*
T0*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ľ
gradients/add_4_grad/Sum_1Sum$gradients/Sigmoid_4_grad/SigmoidGrad,gradients/add_4_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:

gradients/add_4_grad/Reshape_1Reshapegradients/add_4_grad/Sum_1gradients/add_4_grad/Shape_1*
T0*
Tshape0*
_output_shapes	
:
m
%gradients/add_4_grad/tuple/group_depsNoOp^gradients/add_4_grad/Reshape^gradients/add_4_grad/Reshape_1
ă
-gradients/add_4_grad/tuple/control_dependencyIdentitygradients/add_4_grad/Reshape&^gradients/add_4_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/add_4_grad/Reshape*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ü
/gradients/add_4_grad/tuple/control_dependency_1Identitygradients/add_4_grad/Reshape_1&^gradients/add_4_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/add_4_grad/Reshape_1*
_output_shapes	
:
Á
gradients/MatMul_4_grad/MatMulMatMul-gradients/add_4_grad/tuple/control_dependencyVariable_8/read*
transpose_b(*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( 
ľ
 gradients/MatMul_4_grad/MatMul_1MatMul	Sigmoid_3-gradients/add_4_grad/tuple/control_dependency*
transpose_b( *
T0* 
_output_shapes
:
*
transpose_a(
t
(gradients/MatMul_4_grad/tuple/group_depsNoOp^gradients/MatMul_4_grad/MatMul!^gradients/MatMul_4_grad/MatMul_1
í
0gradients/MatMul_4_grad/tuple/control_dependencyIdentitygradients/MatMul_4_grad/MatMul)^gradients/MatMul_4_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/MatMul_4_grad/MatMul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ë
2gradients/MatMul_4_grad/tuple/control_dependency_1Identity gradients/MatMul_4_grad/MatMul_1)^gradients/MatMul_4_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients/MatMul_4_grad/MatMul_1* 
_output_shapes
:

Ł
$gradients/Sigmoid_3_grad/SigmoidGradSigmoidGrad	Sigmoid_30gradients/MatMul_4_grad/tuple/control_dependency*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
b
gradients/add_3_grad/ShapeShapeMatMul_3*
T0*
out_type0*
_output_shapes
:
g
gradients/add_3_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
ş
*gradients/add_3_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_3_grad/Shapegradients/add_3_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
ą
gradients/add_3_grad/SumSum$gradients/Sigmoid_3_grad/SigmoidGrad*gradients/add_3_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:

gradients/add_3_grad/ReshapeReshapegradients/add_3_grad/Sumgradients/add_3_grad/Shape*
T0*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ľ
gradients/add_3_grad/Sum_1Sum$gradients/Sigmoid_3_grad/SigmoidGrad,gradients/add_3_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:

gradients/add_3_grad/Reshape_1Reshapegradients/add_3_grad/Sum_1gradients/add_3_grad/Shape_1*
T0*
Tshape0*
_output_shapes	
:
m
%gradients/add_3_grad/tuple/group_depsNoOp^gradients/add_3_grad/Reshape^gradients/add_3_grad/Reshape_1
ă
-gradients/add_3_grad/tuple/control_dependencyIdentitygradients/add_3_grad/Reshape&^gradients/add_3_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/add_3_grad/Reshape*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ü
/gradients/add_3_grad/tuple/control_dependency_1Identitygradients/add_3_grad/Reshape_1&^gradients/add_3_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/add_3_grad/Reshape_1*
_output_shapes	
:
Á
gradients/MatMul_3_grad/MatMulMatMul-gradients/add_3_grad/tuple/control_dependencyVariable_6/read*
transpose_b(*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( 
ľ
 gradients/MatMul_3_grad/MatMul_1MatMul	Sigmoid_2-gradients/add_3_grad/tuple/control_dependency*
transpose_b( *
T0* 
_output_shapes
:
*
transpose_a(
t
(gradients/MatMul_3_grad/tuple/group_depsNoOp^gradients/MatMul_3_grad/MatMul!^gradients/MatMul_3_grad/MatMul_1
í
0gradients/MatMul_3_grad/tuple/control_dependencyIdentitygradients/MatMul_3_grad/MatMul)^gradients/MatMul_3_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/MatMul_3_grad/MatMul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ë
2gradients/MatMul_3_grad/tuple/control_dependency_1Identity gradients/MatMul_3_grad/MatMul_1)^gradients/MatMul_3_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients/MatMul_3_grad/MatMul_1* 
_output_shapes
:

Ł
$gradients/Sigmoid_2_grad/SigmoidGradSigmoidGrad	Sigmoid_20gradients/MatMul_3_grad/tuple/control_dependency*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
b
gradients/add_2_grad/ShapeShapeMatMul_2*
T0*
out_type0*
_output_shapes
:
g
gradients/add_2_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
ş
*gradients/add_2_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_2_grad/Shapegradients/add_2_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
ą
gradients/add_2_grad/SumSum$gradients/Sigmoid_2_grad/SigmoidGrad*gradients/add_2_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:

gradients/add_2_grad/ReshapeReshapegradients/add_2_grad/Sumgradients/add_2_grad/Shape*
T0*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ľ
gradients/add_2_grad/Sum_1Sum$gradients/Sigmoid_2_grad/SigmoidGrad,gradients/add_2_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:

gradients/add_2_grad/Reshape_1Reshapegradients/add_2_grad/Sum_1gradients/add_2_grad/Shape_1*
T0*
Tshape0*
_output_shapes	
:
m
%gradients/add_2_grad/tuple/group_depsNoOp^gradients/add_2_grad/Reshape^gradients/add_2_grad/Reshape_1
ă
-gradients/add_2_grad/tuple/control_dependencyIdentitygradients/add_2_grad/Reshape&^gradients/add_2_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/add_2_grad/Reshape*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ü
/gradients/add_2_grad/tuple/control_dependency_1Identitygradients/add_2_grad/Reshape_1&^gradients/add_2_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/add_2_grad/Reshape_1*
_output_shapes	
:
Á
gradients/MatMul_2_grad/MatMulMatMul-gradients/add_2_grad/tuple/control_dependencyVariable_4/read*
transpose_b(*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( 
ľ
 gradients/MatMul_2_grad/MatMul_1MatMul	Sigmoid_1-gradients/add_2_grad/tuple/control_dependency*
transpose_b( *
T0* 
_output_shapes
:
*
transpose_a(
t
(gradients/MatMul_2_grad/tuple/group_depsNoOp^gradients/MatMul_2_grad/MatMul!^gradients/MatMul_2_grad/MatMul_1
í
0gradients/MatMul_2_grad/tuple/control_dependencyIdentitygradients/MatMul_2_grad/MatMul)^gradients/MatMul_2_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/MatMul_2_grad/MatMul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ë
2gradients/MatMul_2_grad/tuple/control_dependency_1Identity gradients/MatMul_2_grad/MatMul_1)^gradients/MatMul_2_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients/MatMul_2_grad/MatMul_1* 
_output_shapes
:

Ł
$gradients/Sigmoid_1_grad/SigmoidGradSigmoidGrad	Sigmoid_10gradients/MatMul_2_grad/tuple/control_dependency*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
b
gradients/add_1_grad/ShapeShapeMatMul_1*
T0*
out_type0*
_output_shapes
:
g
gradients/add_1_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
ş
*gradients/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_1_grad/Shapegradients/add_1_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
ą
gradients/add_1_grad/SumSum$gradients/Sigmoid_1_grad/SigmoidGrad*gradients/add_1_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:

gradients/add_1_grad/ReshapeReshapegradients/add_1_grad/Sumgradients/add_1_grad/Shape*
T0*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ľ
gradients/add_1_grad/Sum_1Sum$gradients/Sigmoid_1_grad/SigmoidGrad,gradients/add_1_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:

gradients/add_1_grad/Reshape_1Reshapegradients/add_1_grad/Sum_1gradients/add_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes	
:
m
%gradients/add_1_grad/tuple/group_depsNoOp^gradients/add_1_grad/Reshape^gradients/add_1_grad/Reshape_1
ă
-gradients/add_1_grad/tuple/control_dependencyIdentitygradients/add_1_grad/Reshape&^gradients/add_1_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/add_1_grad/Reshape*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ü
/gradients/add_1_grad/tuple/control_dependency_1Identitygradients/add_1_grad/Reshape_1&^gradients/add_1_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/add_1_grad/Reshape_1*
_output_shapes	
:
Á
gradients/MatMul_1_grad/MatMulMatMul-gradients/add_1_grad/tuple/control_dependencyVariable_2/read*
transpose_b(*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( 
ł
 gradients/MatMul_1_grad/MatMul_1MatMulSigmoid-gradients/add_1_grad/tuple/control_dependency*
transpose_b( *
T0* 
_output_shapes
:
*
transpose_a(
t
(gradients/MatMul_1_grad/tuple/group_depsNoOp^gradients/MatMul_1_grad/MatMul!^gradients/MatMul_1_grad/MatMul_1
í
0gradients/MatMul_1_grad/tuple/control_dependencyIdentitygradients/MatMul_1_grad/MatMul)^gradients/MatMul_1_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/MatMul_1_grad/MatMul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ë
2gradients/MatMul_1_grad/tuple/control_dependency_1Identity gradients/MatMul_1_grad/MatMul_1)^gradients/MatMul_1_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients/MatMul_1_grad/MatMul_1* 
_output_shapes
:


"gradients/Sigmoid_grad/SigmoidGradSigmoidGradSigmoid0gradients/MatMul_1_grad/tuple/control_dependency*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
^
gradients/add_grad/ShapeShapeMatMul*
T0*
out_type0*
_output_shapes
:
e
gradients/add_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
´
(gradients/add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_grad/Shapegradients/add_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
Ť
gradients/add_grad/SumSum"gradients/Sigmoid_grad/SigmoidGrad(gradients/add_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:

gradients/add_grad/ReshapeReshapegradients/add_grad/Sumgradients/add_grad/Shape*
T0*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ż
gradients/add_grad/Sum_1Sum"gradients/Sigmoid_grad/SigmoidGrad*gradients/add_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:

gradients/add_grad/Reshape_1Reshapegradients/add_grad/Sum_1gradients/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes	
:
g
#gradients/add_grad/tuple/group_depsNoOp^gradients/add_grad/Reshape^gradients/add_grad/Reshape_1
Ű
+gradients/add_grad/tuple/control_dependencyIdentitygradients/add_grad/Reshape$^gradients/add_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/add_grad/Reshape*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ô
-gradients/add_grad/tuple/control_dependency_1Identitygradients/add_grad/Reshape_1$^gradients/add_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/add_grad/Reshape_1*
_output_shapes	
:
ş
gradients/MatMul_grad/MatMulMatMul+gradients/add_grad/tuple/control_dependencyVariable/read*
transpose_b(*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( 
Ž
gradients/MatMul_grad/MatMul_1MatMulmyInput+gradients/add_grad/tuple/control_dependency*
transpose_b( *
T0*
_output_shapes
:	*
transpose_a(
n
&gradients/MatMul_grad/tuple/group_depsNoOp^gradients/MatMul_grad/MatMul^gradients/MatMul_grad/MatMul_1
ä
.gradients/MatMul_grad/tuple/control_dependencyIdentitygradients/MatMul_grad/MatMul'^gradients/MatMul_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/MatMul_grad/MatMul*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
â
0gradients/MatMul_grad/tuple/control_dependency_1Identitygradients/MatMul_grad/MatMul_1'^gradients/MatMul_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/MatMul_grad/MatMul_1*
_output_shapes
:	
{
beta1_power/initial_valueConst*
_class
loc:@Variable*
valueB
 *fff?*
dtype0*
_output_shapes
: 

beta1_power
VariableV2*
shared_name *
_class
loc:@Variable*
	container *
shape: *
dtype0*
_output_shapes
: 
Ť
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*
use_locking(*
T0*
_class
loc:@Variable*
validate_shape(*
_output_shapes
: 
g
beta1_power/readIdentitybeta1_power*
T0*
_class
loc:@Variable*
_output_shapes
: 
{
beta2_power/initial_valueConst*
_class
loc:@Variable*
valueB
 *wž?*
dtype0*
_output_shapes
: 

beta2_power
VariableV2*
shared_name *
_class
loc:@Variable*
	container *
shape: *
dtype0*
_output_shapes
: 
Ť
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
use_locking(*
T0*
_class
loc:@Variable*
validate_shape(*
_output_shapes
: 
g
beta2_power/readIdentitybeta2_power*
T0*
_class
loc:@Variable*
_output_shapes
: 

/Variable/Adam/Initializer/zeros/shape_as_tensorConst*
_class
loc:@Variable*
valueB"      *
dtype0*
_output_shapes
:

%Variable/Adam/Initializer/zeros/ConstConst*
_class
loc:@Variable*
valueB
 *    *
dtype0*
_output_shapes
: 
Ř
Variable/Adam/Initializer/zerosFill/Variable/Adam/Initializer/zeros/shape_as_tensor%Variable/Adam/Initializer/zeros/Const*
T0*
_class
loc:@Variable*

index_type0*
_output_shapes
:	
 
Variable/Adam
VariableV2*
shared_name *
_class
loc:@Variable*
	container *
shape:	*
dtype0*
_output_shapes
:	
ž
Variable/Adam/AssignAssignVariable/AdamVariable/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable*
validate_shape(*
_output_shapes
:	
t
Variable/Adam/readIdentityVariable/Adam*
T0*
_class
loc:@Variable*
_output_shapes
:	

1Variable/Adam_1/Initializer/zeros/shape_as_tensorConst*
_class
loc:@Variable*
valueB"      *
dtype0*
_output_shapes
:

'Variable/Adam_1/Initializer/zeros/ConstConst*
_class
loc:@Variable*
valueB
 *    *
dtype0*
_output_shapes
: 
Ţ
!Variable/Adam_1/Initializer/zerosFill1Variable/Adam_1/Initializer/zeros/shape_as_tensor'Variable/Adam_1/Initializer/zeros/Const*
T0*
_class
loc:@Variable*

index_type0*
_output_shapes
:	
˘
Variable/Adam_1
VariableV2*
shared_name *
_class
loc:@Variable*
	container *
shape:	*
dtype0*
_output_shapes
:	
Ä
Variable/Adam_1/AssignAssignVariable/Adam_1!Variable/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable*
validate_shape(*
_output_shapes
:	
x
Variable/Adam_1/readIdentityVariable/Adam_1*
T0*
_class
loc:@Variable*
_output_shapes
:	

!Variable_1/Adam/Initializer/zerosConst*
_class
loc:@Variable_1*
valueB*    *
dtype0*
_output_shapes	
:

Variable_1/Adam
VariableV2*
shared_name *
_class
loc:@Variable_1*
	container *
shape:*
dtype0*
_output_shapes	
:
Â
Variable_1/Adam/AssignAssignVariable_1/Adam!Variable_1/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_1*
validate_shape(*
_output_shapes	
:
v
Variable_1/Adam/readIdentityVariable_1/Adam*
T0*
_class
loc:@Variable_1*
_output_shapes	
:

#Variable_1/Adam_1/Initializer/zerosConst*
_class
loc:@Variable_1*
valueB*    *
dtype0*
_output_shapes	
:

Variable_1/Adam_1
VariableV2*
shared_name *
_class
loc:@Variable_1*
	container *
shape:*
dtype0*
_output_shapes	
:
Č
Variable_1/Adam_1/AssignAssignVariable_1/Adam_1#Variable_1/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_1*
validate_shape(*
_output_shapes	
:
z
Variable_1/Adam_1/readIdentityVariable_1/Adam_1*
T0*
_class
loc:@Variable_1*
_output_shapes	
:
Ą
1Variable_2/Adam/Initializer/zeros/shape_as_tensorConst*
_class
loc:@Variable_2*
valueB"      *
dtype0*
_output_shapes
:

'Variable_2/Adam/Initializer/zeros/ConstConst*
_class
loc:@Variable_2*
valueB
 *    *
dtype0*
_output_shapes
: 
á
!Variable_2/Adam/Initializer/zerosFill1Variable_2/Adam/Initializer/zeros/shape_as_tensor'Variable_2/Adam/Initializer/zeros/Const*
T0*
_class
loc:@Variable_2*

index_type0* 
_output_shapes
:

Ś
Variable_2/Adam
VariableV2*
shared_name *
_class
loc:@Variable_2*
	container *
shape:
*
dtype0* 
_output_shapes
:

Ç
Variable_2/Adam/AssignAssignVariable_2/Adam!Variable_2/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_2*
validate_shape(* 
_output_shapes
:

{
Variable_2/Adam/readIdentityVariable_2/Adam*
T0*
_class
loc:@Variable_2* 
_output_shapes
:

Ł
3Variable_2/Adam_1/Initializer/zeros/shape_as_tensorConst*
_class
loc:@Variable_2*
valueB"      *
dtype0*
_output_shapes
:

)Variable_2/Adam_1/Initializer/zeros/ConstConst*
_class
loc:@Variable_2*
valueB
 *    *
dtype0*
_output_shapes
: 
ç
#Variable_2/Adam_1/Initializer/zerosFill3Variable_2/Adam_1/Initializer/zeros/shape_as_tensor)Variable_2/Adam_1/Initializer/zeros/Const*
T0*
_class
loc:@Variable_2*

index_type0* 
_output_shapes
:

¨
Variable_2/Adam_1
VariableV2*
shared_name *
_class
loc:@Variable_2*
	container *
shape:
*
dtype0* 
_output_shapes
:

Í
Variable_2/Adam_1/AssignAssignVariable_2/Adam_1#Variable_2/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_2*
validate_shape(* 
_output_shapes
:


Variable_2/Adam_1/readIdentityVariable_2/Adam_1*
T0*
_class
loc:@Variable_2* 
_output_shapes
:


!Variable_3/Adam/Initializer/zerosConst*
_class
loc:@Variable_3*
valueB*    *
dtype0*
_output_shapes	
:

Variable_3/Adam
VariableV2*
shared_name *
_class
loc:@Variable_3*
	container *
shape:*
dtype0*
_output_shapes	
:
Â
Variable_3/Adam/AssignAssignVariable_3/Adam!Variable_3/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_3*
validate_shape(*
_output_shapes	
:
v
Variable_3/Adam/readIdentityVariable_3/Adam*
T0*
_class
loc:@Variable_3*
_output_shapes	
:

#Variable_3/Adam_1/Initializer/zerosConst*
_class
loc:@Variable_3*
valueB*    *
dtype0*
_output_shapes	
:

Variable_3/Adam_1
VariableV2*
shared_name *
_class
loc:@Variable_3*
	container *
shape:*
dtype0*
_output_shapes	
:
Č
Variable_3/Adam_1/AssignAssignVariable_3/Adam_1#Variable_3/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_3*
validate_shape(*
_output_shapes	
:
z
Variable_3/Adam_1/readIdentityVariable_3/Adam_1*
T0*
_class
loc:@Variable_3*
_output_shapes	
:
Ą
1Variable_4/Adam/Initializer/zeros/shape_as_tensorConst*
_class
loc:@Variable_4*
valueB"      *
dtype0*
_output_shapes
:

'Variable_4/Adam/Initializer/zeros/ConstConst*
_class
loc:@Variable_4*
valueB
 *    *
dtype0*
_output_shapes
: 
á
!Variable_4/Adam/Initializer/zerosFill1Variable_4/Adam/Initializer/zeros/shape_as_tensor'Variable_4/Adam/Initializer/zeros/Const*
T0*
_class
loc:@Variable_4*

index_type0* 
_output_shapes
:

Ś
Variable_4/Adam
VariableV2*
shared_name *
_class
loc:@Variable_4*
	container *
shape:
*
dtype0* 
_output_shapes
:

Ç
Variable_4/Adam/AssignAssignVariable_4/Adam!Variable_4/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_4*
validate_shape(* 
_output_shapes
:

{
Variable_4/Adam/readIdentityVariable_4/Adam*
T0*
_class
loc:@Variable_4* 
_output_shapes
:

Ł
3Variable_4/Adam_1/Initializer/zeros/shape_as_tensorConst*
_class
loc:@Variable_4*
valueB"      *
dtype0*
_output_shapes
:

)Variable_4/Adam_1/Initializer/zeros/ConstConst*
_class
loc:@Variable_4*
valueB
 *    *
dtype0*
_output_shapes
: 
ç
#Variable_4/Adam_1/Initializer/zerosFill3Variable_4/Adam_1/Initializer/zeros/shape_as_tensor)Variable_4/Adam_1/Initializer/zeros/Const*
T0*
_class
loc:@Variable_4*

index_type0* 
_output_shapes
:

¨
Variable_4/Adam_1
VariableV2*
shared_name *
_class
loc:@Variable_4*
	container *
shape:
*
dtype0* 
_output_shapes
:

Í
Variable_4/Adam_1/AssignAssignVariable_4/Adam_1#Variable_4/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_4*
validate_shape(* 
_output_shapes
:


Variable_4/Adam_1/readIdentityVariable_4/Adam_1*
T0*
_class
loc:@Variable_4* 
_output_shapes
:


!Variable_5/Adam/Initializer/zerosConst*
_class
loc:@Variable_5*
valueB*    *
dtype0*
_output_shapes	
:

Variable_5/Adam
VariableV2*
shared_name *
_class
loc:@Variable_5*
	container *
shape:*
dtype0*
_output_shapes	
:
Â
Variable_5/Adam/AssignAssignVariable_5/Adam!Variable_5/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_5*
validate_shape(*
_output_shapes	
:
v
Variable_5/Adam/readIdentityVariable_5/Adam*
T0*
_class
loc:@Variable_5*
_output_shapes	
:

#Variable_5/Adam_1/Initializer/zerosConst*
_class
loc:@Variable_5*
valueB*    *
dtype0*
_output_shapes	
:

Variable_5/Adam_1
VariableV2*
shared_name *
_class
loc:@Variable_5*
	container *
shape:*
dtype0*
_output_shapes	
:
Č
Variable_5/Adam_1/AssignAssignVariable_5/Adam_1#Variable_5/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_5*
validate_shape(*
_output_shapes	
:
z
Variable_5/Adam_1/readIdentityVariable_5/Adam_1*
T0*
_class
loc:@Variable_5*
_output_shapes	
:
Ą
1Variable_6/Adam/Initializer/zeros/shape_as_tensorConst*
_class
loc:@Variable_6*
valueB"      *
dtype0*
_output_shapes
:

'Variable_6/Adam/Initializer/zeros/ConstConst*
_class
loc:@Variable_6*
valueB
 *    *
dtype0*
_output_shapes
: 
á
!Variable_6/Adam/Initializer/zerosFill1Variable_6/Adam/Initializer/zeros/shape_as_tensor'Variable_6/Adam/Initializer/zeros/Const*
T0*
_class
loc:@Variable_6*

index_type0* 
_output_shapes
:

Ś
Variable_6/Adam
VariableV2*
shared_name *
_class
loc:@Variable_6*
	container *
shape:
*
dtype0* 
_output_shapes
:

Ç
Variable_6/Adam/AssignAssignVariable_6/Adam!Variable_6/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_6*
validate_shape(* 
_output_shapes
:

{
Variable_6/Adam/readIdentityVariable_6/Adam*
T0*
_class
loc:@Variable_6* 
_output_shapes
:

Ł
3Variable_6/Adam_1/Initializer/zeros/shape_as_tensorConst*
_class
loc:@Variable_6*
valueB"      *
dtype0*
_output_shapes
:

)Variable_6/Adam_1/Initializer/zeros/ConstConst*
_class
loc:@Variable_6*
valueB
 *    *
dtype0*
_output_shapes
: 
ç
#Variable_6/Adam_1/Initializer/zerosFill3Variable_6/Adam_1/Initializer/zeros/shape_as_tensor)Variable_6/Adam_1/Initializer/zeros/Const*
T0*
_class
loc:@Variable_6*

index_type0* 
_output_shapes
:

¨
Variable_6/Adam_1
VariableV2*
shared_name *
_class
loc:@Variable_6*
	container *
shape:
*
dtype0* 
_output_shapes
:

Í
Variable_6/Adam_1/AssignAssignVariable_6/Adam_1#Variable_6/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_6*
validate_shape(* 
_output_shapes
:


Variable_6/Adam_1/readIdentityVariable_6/Adam_1*
T0*
_class
loc:@Variable_6* 
_output_shapes
:


!Variable_7/Adam/Initializer/zerosConst*
_class
loc:@Variable_7*
valueB*    *
dtype0*
_output_shapes	
:

Variable_7/Adam
VariableV2*
shared_name *
_class
loc:@Variable_7*
	container *
shape:*
dtype0*
_output_shapes	
:
Â
Variable_7/Adam/AssignAssignVariable_7/Adam!Variable_7/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_7*
validate_shape(*
_output_shapes	
:
v
Variable_7/Adam/readIdentityVariable_7/Adam*
T0*
_class
loc:@Variable_7*
_output_shapes	
:

#Variable_7/Adam_1/Initializer/zerosConst*
_class
loc:@Variable_7*
valueB*    *
dtype0*
_output_shapes	
:

Variable_7/Adam_1
VariableV2*
shared_name *
_class
loc:@Variable_7*
	container *
shape:*
dtype0*
_output_shapes	
:
Č
Variable_7/Adam_1/AssignAssignVariable_7/Adam_1#Variable_7/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_7*
validate_shape(*
_output_shapes	
:
z
Variable_7/Adam_1/readIdentityVariable_7/Adam_1*
T0*
_class
loc:@Variable_7*
_output_shapes	
:
Ą
1Variable_8/Adam/Initializer/zeros/shape_as_tensorConst*
_class
loc:@Variable_8*
valueB"      *
dtype0*
_output_shapes
:

'Variable_8/Adam/Initializer/zeros/ConstConst*
_class
loc:@Variable_8*
valueB
 *    *
dtype0*
_output_shapes
: 
á
!Variable_8/Adam/Initializer/zerosFill1Variable_8/Adam/Initializer/zeros/shape_as_tensor'Variable_8/Adam/Initializer/zeros/Const*
T0*
_class
loc:@Variable_8*

index_type0* 
_output_shapes
:

Ś
Variable_8/Adam
VariableV2*
shared_name *
_class
loc:@Variable_8*
	container *
shape:
*
dtype0* 
_output_shapes
:

Ç
Variable_8/Adam/AssignAssignVariable_8/Adam!Variable_8/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_8*
validate_shape(* 
_output_shapes
:

{
Variable_8/Adam/readIdentityVariable_8/Adam*
T0*
_class
loc:@Variable_8* 
_output_shapes
:

Ł
3Variable_8/Adam_1/Initializer/zeros/shape_as_tensorConst*
_class
loc:@Variable_8*
valueB"      *
dtype0*
_output_shapes
:

)Variable_8/Adam_1/Initializer/zeros/ConstConst*
_class
loc:@Variable_8*
valueB
 *    *
dtype0*
_output_shapes
: 
ç
#Variable_8/Adam_1/Initializer/zerosFill3Variable_8/Adam_1/Initializer/zeros/shape_as_tensor)Variable_8/Adam_1/Initializer/zeros/Const*
T0*
_class
loc:@Variable_8*

index_type0* 
_output_shapes
:

¨
Variable_8/Adam_1
VariableV2*
shared_name *
_class
loc:@Variable_8*
	container *
shape:
*
dtype0* 
_output_shapes
:

Í
Variable_8/Adam_1/AssignAssignVariable_8/Adam_1#Variable_8/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_8*
validate_shape(* 
_output_shapes
:


Variable_8/Adam_1/readIdentityVariable_8/Adam_1*
T0*
_class
loc:@Variable_8* 
_output_shapes
:


!Variable_9/Adam/Initializer/zerosConst*
_class
loc:@Variable_9*
valueB*    *
dtype0*
_output_shapes	
:

Variable_9/Adam
VariableV2*
shared_name *
_class
loc:@Variable_9*
	container *
shape:*
dtype0*
_output_shapes	
:
Â
Variable_9/Adam/AssignAssignVariable_9/Adam!Variable_9/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_9*
validate_shape(*
_output_shapes	
:
v
Variable_9/Adam/readIdentityVariable_9/Adam*
T0*
_class
loc:@Variable_9*
_output_shapes	
:

#Variable_9/Adam_1/Initializer/zerosConst*
_class
loc:@Variable_9*
valueB*    *
dtype0*
_output_shapes	
:

Variable_9/Adam_1
VariableV2*
shared_name *
_class
loc:@Variable_9*
	container *
shape:*
dtype0*
_output_shapes	
:
Č
Variable_9/Adam_1/AssignAssignVariable_9/Adam_1#Variable_9/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_9*
validate_shape(*
_output_shapes	
:
z
Variable_9/Adam_1/readIdentityVariable_9/Adam_1*
T0*
_class
loc:@Variable_9*
_output_shapes	
:
Ł
2Variable_10/Adam/Initializer/zeros/shape_as_tensorConst*
_class
loc:@Variable_10*
valueB"      *
dtype0*
_output_shapes
:

(Variable_10/Adam/Initializer/zeros/ConstConst*
_class
loc:@Variable_10*
valueB
 *    *
dtype0*
_output_shapes
: 
ĺ
"Variable_10/Adam/Initializer/zerosFill2Variable_10/Adam/Initializer/zeros/shape_as_tensor(Variable_10/Adam/Initializer/zeros/Const*
T0*
_class
loc:@Variable_10*

index_type0* 
_output_shapes
:

¨
Variable_10/Adam
VariableV2*
shared_name *
_class
loc:@Variable_10*
	container *
shape:
*
dtype0* 
_output_shapes
:

Ë
Variable_10/Adam/AssignAssignVariable_10/Adam"Variable_10/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_10*
validate_shape(* 
_output_shapes
:

~
Variable_10/Adam/readIdentityVariable_10/Adam*
T0*
_class
loc:@Variable_10* 
_output_shapes
:

Ľ
4Variable_10/Adam_1/Initializer/zeros/shape_as_tensorConst*
_class
loc:@Variable_10*
valueB"      *
dtype0*
_output_shapes
:

*Variable_10/Adam_1/Initializer/zeros/ConstConst*
_class
loc:@Variable_10*
valueB
 *    *
dtype0*
_output_shapes
: 
ë
$Variable_10/Adam_1/Initializer/zerosFill4Variable_10/Adam_1/Initializer/zeros/shape_as_tensor*Variable_10/Adam_1/Initializer/zeros/Const*
T0*
_class
loc:@Variable_10*

index_type0* 
_output_shapes
:

Ş
Variable_10/Adam_1
VariableV2*
shared_name *
_class
loc:@Variable_10*
	container *
shape:
*
dtype0* 
_output_shapes
:

Ń
Variable_10/Adam_1/AssignAssignVariable_10/Adam_1$Variable_10/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_10*
validate_shape(* 
_output_shapes
:


Variable_10/Adam_1/readIdentityVariable_10/Adam_1*
T0*
_class
loc:@Variable_10* 
_output_shapes
:


"Variable_11/Adam/Initializer/zerosConst*
_class
loc:@Variable_11*
valueB*    *
dtype0*
_output_shapes	
:

Variable_11/Adam
VariableV2*
shared_name *
_class
loc:@Variable_11*
	container *
shape:*
dtype0*
_output_shapes	
:
Ć
Variable_11/Adam/AssignAssignVariable_11/Adam"Variable_11/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_11*
validate_shape(*
_output_shapes	
:
y
Variable_11/Adam/readIdentityVariable_11/Adam*
T0*
_class
loc:@Variable_11*
_output_shapes	
:

$Variable_11/Adam_1/Initializer/zerosConst*
_class
loc:@Variable_11*
valueB*    *
dtype0*
_output_shapes	
:
 
Variable_11/Adam_1
VariableV2*
shared_name *
_class
loc:@Variable_11*
	container *
shape:*
dtype0*
_output_shapes	
:
Ě
Variable_11/Adam_1/AssignAssignVariable_11/Adam_1$Variable_11/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_11*
validate_shape(*
_output_shapes	
:
}
Variable_11/Adam_1/readIdentityVariable_11/Adam_1*
T0*
_class
loc:@Variable_11*
_output_shapes	
:
Ł
2Variable_12/Adam/Initializer/zeros/shape_as_tensorConst*
_class
loc:@Variable_12*
valueB"      *
dtype0*
_output_shapes
:

(Variable_12/Adam/Initializer/zeros/ConstConst*
_class
loc:@Variable_12*
valueB
 *    *
dtype0*
_output_shapes
: 
ĺ
"Variable_12/Adam/Initializer/zerosFill2Variable_12/Adam/Initializer/zeros/shape_as_tensor(Variable_12/Adam/Initializer/zeros/Const*
T0*
_class
loc:@Variable_12*

index_type0* 
_output_shapes
:

¨
Variable_12/Adam
VariableV2*
shared_name *
_class
loc:@Variable_12*
	container *
shape:
*
dtype0* 
_output_shapes
:

Ë
Variable_12/Adam/AssignAssignVariable_12/Adam"Variable_12/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_12*
validate_shape(* 
_output_shapes
:

~
Variable_12/Adam/readIdentityVariable_12/Adam*
T0*
_class
loc:@Variable_12* 
_output_shapes
:

Ľ
4Variable_12/Adam_1/Initializer/zeros/shape_as_tensorConst*
_class
loc:@Variable_12*
valueB"      *
dtype0*
_output_shapes
:

*Variable_12/Adam_1/Initializer/zeros/ConstConst*
_class
loc:@Variable_12*
valueB
 *    *
dtype0*
_output_shapes
: 
ë
$Variable_12/Adam_1/Initializer/zerosFill4Variable_12/Adam_1/Initializer/zeros/shape_as_tensor*Variable_12/Adam_1/Initializer/zeros/Const*
T0*
_class
loc:@Variable_12*

index_type0* 
_output_shapes
:

Ş
Variable_12/Adam_1
VariableV2*
shared_name *
_class
loc:@Variable_12*
	container *
shape:
*
dtype0* 
_output_shapes
:

Ń
Variable_12/Adam_1/AssignAssignVariable_12/Adam_1$Variable_12/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_12*
validate_shape(* 
_output_shapes
:


Variable_12/Adam_1/readIdentityVariable_12/Adam_1*
T0*
_class
loc:@Variable_12* 
_output_shapes
:


"Variable_13/Adam/Initializer/zerosConst*
_class
loc:@Variable_13*
valueB*    *
dtype0*
_output_shapes	
:

Variable_13/Adam
VariableV2*
shared_name *
_class
loc:@Variable_13*
	container *
shape:*
dtype0*
_output_shapes	
:
Ć
Variable_13/Adam/AssignAssignVariable_13/Adam"Variable_13/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_13*
validate_shape(*
_output_shapes	
:
y
Variable_13/Adam/readIdentityVariable_13/Adam*
T0*
_class
loc:@Variable_13*
_output_shapes	
:

$Variable_13/Adam_1/Initializer/zerosConst*
_class
loc:@Variable_13*
valueB*    *
dtype0*
_output_shapes	
:
 
Variable_13/Adam_1
VariableV2*
shared_name *
_class
loc:@Variable_13*
	container *
shape:*
dtype0*
_output_shapes	
:
Ě
Variable_13/Adam_1/AssignAssignVariable_13/Adam_1$Variable_13/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_13*
validate_shape(*
_output_shapes	
:
}
Variable_13/Adam_1/readIdentityVariable_13/Adam_1*
T0*
_class
loc:@Variable_13*
_output_shapes	
:
Ł
2Variable_14/Adam/Initializer/zeros/shape_as_tensorConst*
_class
loc:@Variable_14*
valueB"      *
dtype0*
_output_shapes
:

(Variable_14/Adam/Initializer/zeros/ConstConst*
_class
loc:@Variable_14*
valueB
 *    *
dtype0*
_output_shapes
: 
ĺ
"Variable_14/Adam/Initializer/zerosFill2Variable_14/Adam/Initializer/zeros/shape_as_tensor(Variable_14/Adam/Initializer/zeros/Const*
T0*
_class
loc:@Variable_14*

index_type0* 
_output_shapes
:

¨
Variable_14/Adam
VariableV2*
shared_name *
_class
loc:@Variable_14*
	container *
shape:
*
dtype0* 
_output_shapes
:

Ë
Variable_14/Adam/AssignAssignVariable_14/Adam"Variable_14/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_14*
validate_shape(* 
_output_shapes
:

~
Variable_14/Adam/readIdentityVariable_14/Adam*
T0*
_class
loc:@Variable_14* 
_output_shapes
:

Ľ
4Variable_14/Adam_1/Initializer/zeros/shape_as_tensorConst*
_class
loc:@Variable_14*
valueB"      *
dtype0*
_output_shapes
:

*Variable_14/Adam_1/Initializer/zeros/ConstConst*
_class
loc:@Variable_14*
valueB
 *    *
dtype0*
_output_shapes
: 
ë
$Variable_14/Adam_1/Initializer/zerosFill4Variable_14/Adam_1/Initializer/zeros/shape_as_tensor*Variable_14/Adam_1/Initializer/zeros/Const*
T0*
_class
loc:@Variable_14*

index_type0* 
_output_shapes
:

Ş
Variable_14/Adam_1
VariableV2*
shared_name *
_class
loc:@Variable_14*
	container *
shape:
*
dtype0* 
_output_shapes
:

Ń
Variable_14/Adam_1/AssignAssignVariable_14/Adam_1$Variable_14/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_14*
validate_shape(* 
_output_shapes
:


Variable_14/Adam_1/readIdentityVariable_14/Adam_1*
T0*
_class
loc:@Variable_14* 
_output_shapes
:


"Variable_15/Adam/Initializer/zerosConst*
_class
loc:@Variable_15*
valueB*    *
dtype0*
_output_shapes	
:

Variable_15/Adam
VariableV2*
shared_name *
_class
loc:@Variable_15*
	container *
shape:*
dtype0*
_output_shapes	
:
Ć
Variable_15/Adam/AssignAssignVariable_15/Adam"Variable_15/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_15*
validate_shape(*
_output_shapes	
:
y
Variable_15/Adam/readIdentityVariable_15/Adam*
T0*
_class
loc:@Variable_15*
_output_shapes	
:

$Variable_15/Adam_1/Initializer/zerosConst*
_class
loc:@Variable_15*
valueB*    *
dtype0*
_output_shapes	
:
 
Variable_15/Adam_1
VariableV2*
shared_name *
_class
loc:@Variable_15*
	container *
shape:*
dtype0*
_output_shapes	
:
Ě
Variable_15/Adam_1/AssignAssignVariable_15/Adam_1$Variable_15/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_15*
validate_shape(*
_output_shapes	
:
}
Variable_15/Adam_1/readIdentityVariable_15/Adam_1*
T0*
_class
loc:@Variable_15*
_output_shapes	
:

"Variable_16/Adam/Initializer/zerosConst*
_class
loc:@Variable_16*
valueB	*    *
dtype0*
_output_shapes
:	
Ś
Variable_16/Adam
VariableV2*
shared_name *
_class
loc:@Variable_16*
	container *
shape:	*
dtype0*
_output_shapes
:	
Ę
Variable_16/Adam/AssignAssignVariable_16/Adam"Variable_16/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_16*
validate_shape(*
_output_shapes
:	
}
Variable_16/Adam/readIdentityVariable_16/Adam*
T0*
_class
loc:@Variable_16*
_output_shapes
:	

$Variable_16/Adam_1/Initializer/zerosConst*
_class
loc:@Variable_16*
valueB	*    *
dtype0*
_output_shapes
:	
¨
Variable_16/Adam_1
VariableV2*
shared_name *
_class
loc:@Variable_16*
	container *
shape:	*
dtype0*
_output_shapes
:	
Đ
Variable_16/Adam_1/AssignAssignVariable_16/Adam_1$Variable_16/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_16*
validate_shape(*
_output_shapes
:	

Variable_16/Adam_1/readIdentityVariable_16/Adam_1*
T0*
_class
loc:@Variable_16*
_output_shapes
:	

"Variable_17/Adam/Initializer/zerosConst*
_class
loc:@Variable_17*
valueB*    *
dtype0*
_output_shapes
:

Variable_17/Adam
VariableV2*
shared_name *
_class
loc:@Variable_17*
	container *
shape:*
dtype0*
_output_shapes
:
Ĺ
Variable_17/Adam/AssignAssignVariable_17/Adam"Variable_17/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_17*
validate_shape(*
_output_shapes
:
x
Variable_17/Adam/readIdentityVariable_17/Adam*
T0*
_class
loc:@Variable_17*
_output_shapes
:

$Variable_17/Adam_1/Initializer/zerosConst*
_class
loc:@Variable_17*
valueB*    *
dtype0*
_output_shapes
:

Variable_17/Adam_1
VariableV2*
shared_name *
_class
loc:@Variable_17*
	container *
shape:*
dtype0*
_output_shapes
:
Ë
Variable_17/Adam_1/AssignAssignVariable_17/Adam_1$Variable_17/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_17*
validate_shape(*
_output_shapes
:
|
Variable_17/Adam_1/readIdentityVariable_17/Adam_1*
T0*
_class
loc:@Variable_17*
_output_shapes
:

"Variable_18/Adam/Initializer/zerosConst*
_class
loc:@Variable_18*
valueB*    *
dtype0*
_output_shapes

:
¤
Variable_18/Adam
VariableV2*
shared_name *
_class
loc:@Variable_18*
	container *
shape
:*
dtype0*
_output_shapes

:
É
Variable_18/Adam/AssignAssignVariable_18/Adam"Variable_18/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_18*
validate_shape(*
_output_shapes

:
|
Variable_18/Adam/readIdentityVariable_18/Adam*
T0*
_class
loc:@Variable_18*
_output_shapes

:

$Variable_18/Adam_1/Initializer/zerosConst*
_class
loc:@Variable_18*
valueB*    *
dtype0*
_output_shapes

:
Ś
Variable_18/Adam_1
VariableV2*
shared_name *
_class
loc:@Variable_18*
	container *
shape
:*
dtype0*
_output_shapes

:
Ď
Variable_18/Adam_1/AssignAssignVariable_18/Adam_1$Variable_18/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_18*
validate_shape(*
_output_shapes

:

Variable_18/Adam_1/readIdentityVariable_18/Adam_1*
T0*
_class
loc:@Variable_18*
_output_shapes

:

"Variable_19/Adam/Initializer/zerosConst*
_class
loc:@Variable_19*
valueB*    *
dtype0*
_output_shapes
:

Variable_19/Adam
VariableV2*
shared_name *
_class
loc:@Variable_19*
	container *
shape:*
dtype0*
_output_shapes
:
Ĺ
Variable_19/Adam/AssignAssignVariable_19/Adam"Variable_19/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_19*
validate_shape(*
_output_shapes
:
x
Variable_19/Adam/readIdentityVariable_19/Adam*
T0*
_class
loc:@Variable_19*
_output_shapes
:

$Variable_19/Adam_1/Initializer/zerosConst*
_class
loc:@Variable_19*
valueB*    *
dtype0*
_output_shapes
:

Variable_19/Adam_1
VariableV2*
shared_name *
_class
loc:@Variable_19*
	container *
shape:*
dtype0*
_output_shapes
:
Ë
Variable_19/Adam_1/AssignAssignVariable_19/Adam_1$Variable_19/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_19*
validate_shape(*
_output_shapes
:
|
Variable_19/Adam_1/readIdentityVariable_19/Adam_1*
T0*
_class
loc:@Variable_19*
_output_shapes
:
W
Adam/learning_rateConst*
valueB
 *o:*
dtype0*
_output_shapes
: 
O

Adam/beta1Const*
valueB
 *fff?*
dtype0*
_output_shapes
: 
O

Adam/beta2Const*
valueB
 *wž?*
dtype0*
_output_shapes
: 
Q
Adam/epsilonConst*
valueB
 *wĚ+2*
dtype0*
_output_shapes
: 
Ó
Adam/update_Variable/ApplyAdam	ApplyAdamVariableVariable/AdamVariable/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon0gradients/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@Variable*
use_nesterov( *
_output_shapes
:	
Ö
 Adam/update_Variable_1/ApplyAdam	ApplyAdam
Variable_1Variable_1/AdamVariable_1/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon-gradients/add_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@Variable_1*
use_nesterov( *
_output_shapes	
:
ŕ
 Adam/update_Variable_2/ApplyAdam	ApplyAdam
Variable_2Variable_2/AdamVariable_2/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/MatMul_1_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@Variable_2*
use_nesterov( * 
_output_shapes
:

Ř
 Adam/update_Variable_3/ApplyAdam	ApplyAdam
Variable_3Variable_3/AdamVariable_3/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon/gradients/add_1_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@Variable_3*
use_nesterov( *
_output_shapes	
:
ŕ
 Adam/update_Variable_4/ApplyAdam	ApplyAdam
Variable_4Variable_4/AdamVariable_4/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/MatMul_2_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@Variable_4*
use_nesterov( * 
_output_shapes
:

Ř
 Adam/update_Variable_5/ApplyAdam	ApplyAdam
Variable_5Variable_5/AdamVariable_5/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon/gradients/add_2_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@Variable_5*
use_nesterov( *
_output_shapes	
:
ŕ
 Adam/update_Variable_6/ApplyAdam	ApplyAdam
Variable_6Variable_6/AdamVariable_6/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/MatMul_3_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@Variable_6*
use_nesterov( * 
_output_shapes
:

Ř
 Adam/update_Variable_7/ApplyAdam	ApplyAdam
Variable_7Variable_7/AdamVariable_7/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon/gradients/add_3_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@Variable_7*
use_nesterov( *
_output_shapes	
:
ŕ
 Adam/update_Variable_8/ApplyAdam	ApplyAdam
Variable_8Variable_8/AdamVariable_8/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/MatMul_4_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@Variable_8*
use_nesterov( * 
_output_shapes
:

Ř
 Adam/update_Variable_9/ApplyAdam	ApplyAdam
Variable_9Variable_9/AdamVariable_9/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon/gradients/add_4_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@Variable_9*
use_nesterov( *
_output_shapes	
:
ĺ
!Adam/update_Variable_10/ApplyAdam	ApplyAdamVariable_10Variable_10/AdamVariable_10/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/MatMul_5_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@Variable_10*
use_nesterov( * 
_output_shapes
:

Ý
!Adam/update_Variable_11/ApplyAdam	ApplyAdamVariable_11Variable_11/AdamVariable_11/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon/gradients/add_5_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@Variable_11*
use_nesterov( *
_output_shapes	
:
ĺ
!Adam/update_Variable_12/ApplyAdam	ApplyAdamVariable_12Variable_12/AdamVariable_12/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/MatMul_6_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@Variable_12*
use_nesterov( * 
_output_shapes
:

Ý
!Adam/update_Variable_13/ApplyAdam	ApplyAdamVariable_13Variable_13/AdamVariable_13/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon/gradients/add_6_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@Variable_13*
use_nesterov( *
_output_shapes	
:
ĺ
!Adam/update_Variable_14/ApplyAdam	ApplyAdamVariable_14Variable_14/AdamVariable_14/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/MatMul_7_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@Variable_14*
use_nesterov( * 
_output_shapes
:

Ý
!Adam/update_Variable_15/ApplyAdam	ApplyAdamVariable_15Variable_15/AdamVariable_15/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon/gradients/add_7_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@Variable_15*
use_nesterov( *
_output_shapes	
:
ä
!Adam/update_Variable_16/ApplyAdam	ApplyAdamVariable_16Variable_16/AdamVariable_16/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/MatMul_8_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@Variable_16*
use_nesterov( *
_output_shapes
:	
Ü
!Adam/update_Variable_17/ApplyAdam	ApplyAdamVariable_17Variable_17/AdamVariable_17/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon/gradients/add_8_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@Variable_17*
use_nesterov( *
_output_shapes
:
ă
!Adam/update_Variable_18/ApplyAdam	ApplyAdamVariable_18Variable_18/AdamVariable_18/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/myOutput_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@Variable_18*
use_nesterov( *
_output_shapes

:
Ü
!Adam/update_Variable_19/ApplyAdam	ApplyAdamVariable_19Variable_19/AdamVariable_19/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon/gradients/add_9_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@Variable_19*
use_nesterov( *
_output_shapes
:
Ż
Adam/mulMulbeta1_power/read
Adam/beta1^Adam/update_Variable/ApplyAdam!^Adam/update_Variable_1/ApplyAdam"^Adam/update_Variable_10/ApplyAdam"^Adam/update_Variable_11/ApplyAdam"^Adam/update_Variable_12/ApplyAdam"^Adam/update_Variable_13/ApplyAdam"^Adam/update_Variable_14/ApplyAdam"^Adam/update_Variable_15/ApplyAdam"^Adam/update_Variable_16/ApplyAdam"^Adam/update_Variable_17/ApplyAdam"^Adam/update_Variable_18/ApplyAdam"^Adam/update_Variable_19/ApplyAdam!^Adam/update_Variable_2/ApplyAdam!^Adam/update_Variable_3/ApplyAdam!^Adam/update_Variable_4/ApplyAdam!^Adam/update_Variable_5/ApplyAdam!^Adam/update_Variable_6/ApplyAdam!^Adam/update_Variable_7/ApplyAdam!^Adam/update_Variable_8/ApplyAdam!^Adam/update_Variable_9/ApplyAdam*
T0*
_class
loc:@Variable*
_output_shapes
: 

Adam/AssignAssignbeta1_powerAdam/mul*
use_locking( *
T0*
_class
loc:@Variable*
validate_shape(*
_output_shapes
: 
ą

Adam/mul_1Mulbeta2_power/read
Adam/beta2^Adam/update_Variable/ApplyAdam!^Adam/update_Variable_1/ApplyAdam"^Adam/update_Variable_10/ApplyAdam"^Adam/update_Variable_11/ApplyAdam"^Adam/update_Variable_12/ApplyAdam"^Adam/update_Variable_13/ApplyAdam"^Adam/update_Variable_14/ApplyAdam"^Adam/update_Variable_15/ApplyAdam"^Adam/update_Variable_16/ApplyAdam"^Adam/update_Variable_17/ApplyAdam"^Adam/update_Variable_18/ApplyAdam"^Adam/update_Variable_19/ApplyAdam!^Adam/update_Variable_2/ApplyAdam!^Adam/update_Variable_3/ApplyAdam!^Adam/update_Variable_4/ApplyAdam!^Adam/update_Variable_5/ApplyAdam!^Adam/update_Variable_6/ApplyAdam!^Adam/update_Variable_7/ApplyAdam!^Adam/update_Variable_8/ApplyAdam!^Adam/update_Variable_9/ApplyAdam*
T0*
_class
loc:@Variable*
_output_shapes
: 

Adam/Assign_1Assignbeta2_power
Adam/mul_1*
use_locking( *
T0*
_class
loc:@Variable*
validate_shape(*
_output_shapes
: 
î
AdamNoOp^Adam/Assign^Adam/Assign_1^Adam/update_Variable/ApplyAdam!^Adam/update_Variable_1/ApplyAdam"^Adam/update_Variable_10/ApplyAdam"^Adam/update_Variable_11/ApplyAdam"^Adam/update_Variable_12/ApplyAdam"^Adam/update_Variable_13/ApplyAdam"^Adam/update_Variable_14/ApplyAdam"^Adam/update_Variable_15/ApplyAdam"^Adam/update_Variable_16/ApplyAdam"^Adam/update_Variable_17/ApplyAdam"^Adam/update_Variable_18/ApplyAdam"^Adam/update_Variable_19/ApplyAdam!^Adam/update_Variable_2/ApplyAdam!^Adam/update_Variable_3/ApplyAdam!^Adam/update_Variable_4/ApplyAdam!^Adam/update_Variable_5/ApplyAdam!^Adam/update_Variable_6/ApplyAdam!^Adam/update_Variable_7/ApplyAdam!^Adam/update_Variable_8/ApplyAdam!^Adam/update_Variable_9/ApplyAdam
î
initNoOp^Variable/Adam/Assign^Variable/Adam_1/Assign^Variable/Assign^Variable_1/Adam/Assign^Variable_1/Adam_1/Assign^Variable_1/Assign^Variable_10/Adam/Assign^Variable_10/Adam_1/Assign^Variable_10/Assign^Variable_11/Adam/Assign^Variable_11/Adam_1/Assign^Variable_11/Assign^Variable_12/Adam/Assign^Variable_12/Adam_1/Assign^Variable_12/Assign^Variable_13/Adam/Assign^Variable_13/Adam_1/Assign^Variable_13/Assign^Variable_14/Adam/Assign^Variable_14/Adam_1/Assign^Variable_14/Assign^Variable_15/Adam/Assign^Variable_15/Adam_1/Assign^Variable_15/Assign^Variable_16/Adam/Assign^Variable_16/Adam_1/Assign^Variable_16/Assign^Variable_17/Adam/Assign^Variable_17/Adam_1/Assign^Variable_17/Assign^Variable_18/Adam/Assign^Variable_18/Adam_1/Assign^Variable_18/Assign^Variable_19/Adam/Assign^Variable_19/Adam_1/Assign^Variable_19/Assign^Variable_2/Adam/Assign^Variable_2/Adam_1/Assign^Variable_2/Assign^Variable_3/Adam/Assign^Variable_3/Adam_1/Assign^Variable_3/Assign^Variable_4/Adam/Assign^Variable_4/Adam_1/Assign^Variable_4/Assign^Variable_5/Adam/Assign^Variable_5/Adam_1/Assign^Variable_5/Assign^Variable_6/Adam/Assign^Variable_6/Adam_1/Assign^Variable_6/Assign^Variable_7/Adam/Assign^Variable_7/Adam_1/Assign^Variable_7/Assign^Variable_8/Adam/Assign^Variable_8/Adam_1/Assign^Variable_8/Assign^Variable_9/Adam/Assign^Variable_9/Adam_1/Assign^Variable_9/Assign^beta1_power/Assign^beta2_power/Assign
P
subSubadd_9Placeholder*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
G
SquareSquaresub*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
X
Const_1Const*
valueB"       *
dtype0*
_output_shapes
:
]
Mean_1MeanSquareConst_1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
P

save/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 

save/StringJoin/inputs_1Const*<
value3B1 B+_temp_d8777ceda38143ad9df171c81bf941d2/part*
dtype0*
_output_shapes
: 
u
save/StringJoin
StringJoin
save/Constsave/StringJoin/inputs_1*
	separator *
N*
_output_shapes
: 
Q
save/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
\
save/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 
}
save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards*
_output_shapes
: 
Ô
save/SaveV2/tensor_namesConst*
valueýBú>BVariableBVariable/AdamBVariable/Adam_1B
Variable_1BVariable_1/AdamBVariable_1/Adam_1BVariable_10BVariable_10/AdamBVariable_10/Adam_1BVariable_11BVariable_11/AdamBVariable_11/Adam_1BVariable_12BVariable_12/AdamBVariable_12/Adam_1BVariable_13BVariable_13/AdamBVariable_13/Adam_1BVariable_14BVariable_14/AdamBVariable_14/Adam_1BVariable_15BVariable_15/AdamBVariable_15/Adam_1BVariable_16BVariable_16/AdamBVariable_16/Adam_1BVariable_17BVariable_17/AdamBVariable_17/Adam_1BVariable_18BVariable_18/AdamBVariable_18/Adam_1BVariable_19BVariable_19/AdamBVariable_19/Adam_1B
Variable_2BVariable_2/AdamBVariable_2/Adam_1B
Variable_3BVariable_3/AdamBVariable_3/Adam_1B
Variable_4BVariable_4/AdamBVariable_4/Adam_1B
Variable_5BVariable_5/AdamBVariable_5/Adam_1B
Variable_6BVariable_6/AdamBVariable_6/Adam_1B
Variable_7BVariable_7/AdamBVariable_7/Adam_1B
Variable_8BVariable_8/AdamBVariable_8/Adam_1B
Variable_9BVariable_9/AdamBVariable_9/Adam_1Bbeta1_powerBbeta2_power*
dtype0*
_output_shapes
:>
â
save/SaveV2/shape_and_slicesConst*
valueB>B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:>
Ł	
save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesVariableVariable/AdamVariable/Adam_1
Variable_1Variable_1/AdamVariable_1/Adam_1Variable_10Variable_10/AdamVariable_10/Adam_1Variable_11Variable_11/AdamVariable_11/Adam_1Variable_12Variable_12/AdamVariable_12/Adam_1Variable_13Variable_13/AdamVariable_13/Adam_1Variable_14Variable_14/AdamVariable_14/Adam_1Variable_15Variable_15/AdamVariable_15/Adam_1Variable_16Variable_16/AdamVariable_16/Adam_1Variable_17Variable_17/AdamVariable_17/Adam_1Variable_18Variable_18/AdamVariable_18/Adam_1Variable_19Variable_19/AdamVariable_19/Adam_1
Variable_2Variable_2/AdamVariable_2/Adam_1
Variable_3Variable_3/AdamVariable_3/Adam_1
Variable_4Variable_4/AdamVariable_4/Adam_1
Variable_5Variable_5/AdamVariable_5/Adam_1
Variable_6Variable_6/AdamVariable_6/Adam_1
Variable_7Variable_7/AdamVariable_7/Adam_1
Variable_8Variable_8/AdamVariable_8/Adam_1
Variable_9Variable_9/AdamVariable_9/Adam_1beta1_powerbeta2_power*L
dtypesB
@2>

save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2*
T0*'
_class
loc:@save/ShardedFilename*
_output_shapes
: 

+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilename^save/control_dependency*
T0*

axis *
N*
_output_shapes
:
}
save/MergeV2CheckpointsMergeV2Checkpoints+save/MergeV2Checkpoints/checkpoint_prefixes
save/Const*
delete_old_dirs(
z
save/IdentityIdentity
save/Const^save/MergeV2Checkpoints^save/control_dependency*
T0*
_output_shapes
: 
×
save/RestoreV2/tensor_namesConst*
valueýBú>BVariableBVariable/AdamBVariable/Adam_1B
Variable_1BVariable_1/AdamBVariable_1/Adam_1BVariable_10BVariable_10/AdamBVariable_10/Adam_1BVariable_11BVariable_11/AdamBVariable_11/Adam_1BVariable_12BVariable_12/AdamBVariable_12/Adam_1BVariable_13BVariable_13/AdamBVariable_13/Adam_1BVariable_14BVariable_14/AdamBVariable_14/Adam_1BVariable_15BVariable_15/AdamBVariable_15/Adam_1BVariable_16BVariable_16/AdamBVariable_16/Adam_1BVariable_17BVariable_17/AdamBVariable_17/Adam_1BVariable_18BVariable_18/AdamBVariable_18/Adam_1BVariable_19BVariable_19/AdamBVariable_19/Adam_1B
Variable_2BVariable_2/AdamBVariable_2/Adam_1B
Variable_3BVariable_3/AdamBVariable_3/Adam_1B
Variable_4BVariable_4/AdamBVariable_4/Adam_1B
Variable_5BVariable_5/AdamBVariable_5/Adam_1B
Variable_6BVariable_6/AdamBVariable_6/Adam_1B
Variable_7BVariable_7/AdamBVariable_7/Adam_1B
Variable_8BVariable_8/AdamBVariable_8/Adam_1B
Variable_9BVariable_9/AdamBVariable_9/Adam_1Bbeta1_powerBbeta2_power*
dtype0*
_output_shapes
:>
ĺ
save/RestoreV2/shape_and_slicesConst*
valueB>B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:>
Ä
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices*L
dtypesB
@2>*
_output_shapesű
ř::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

save/AssignAssignVariablesave/RestoreV2*
use_locking(*
T0*
_class
loc:@Variable*
validate_shape(*
_output_shapes
:	
¨
save/Assign_1AssignVariable/Adamsave/RestoreV2:1*
use_locking(*
T0*
_class
loc:@Variable*
validate_shape(*
_output_shapes
:	
Ş
save/Assign_2AssignVariable/Adam_1save/RestoreV2:2*
use_locking(*
T0*
_class
loc:@Variable*
validate_shape(*
_output_shapes
:	
Ł
save/Assign_3Assign
Variable_1save/RestoreV2:3*
use_locking(*
T0*
_class
loc:@Variable_1*
validate_shape(*
_output_shapes	
:
¨
save/Assign_4AssignVariable_1/Adamsave/RestoreV2:4*
use_locking(*
T0*
_class
loc:@Variable_1*
validate_shape(*
_output_shapes	
:
Ş
save/Assign_5AssignVariable_1/Adam_1save/RestoreV2:5*
use_locking(*
T0*
_class
loc:@Variable_1*
validate_shape(*
_output_shapes	
:
Ş
save/Assign_6AssignVariable_10save/RestoreV2:6*
use_locking(*
T0*
_class
loc:@Variable_10*
validate_shape(* 
_output_shapes
:

Ż
save/Assign_7AssignVariable_10/Adamsave/RestoreV2:7*
use_locking(*
T0*
_class
loc:@Variable_10*
validate_shape(* 
_output_shapes
:

ą
save/Assign_8AssignVariable_10/Adam_1save/RestoreV2:8*
use_locking(*
T0*
_class
loc:@Variable_10*
validate_shape(* 
_output_shapes
:

Ľ
save/Assign_9AssignVariable_11save/RestoreV2:9*
use_locking(*
T0*
_class
loc:@Variable_11*
validate_shape(*
_output_shapes	
:
Ź
save/Assign_10AssignVariable_11/Adamsave/RestoreV2:10*
use_locking(*
T0*
_class
loc:@Variable_11*
validate_shape(*
_output_shapes	
:
Ž
save/Assign_11AssignVariable_11/Adam_1save/RestoreV2:11*
use_locking(*
T0*
_class
loc:@Variable_11*
validate_shape(*
_output_shapes	
:
Ź
save/Assign_12AssignVariable_12save/RestoreV2:12*
use_locking(*
T0*
_class
loc:@Variable_12*
validate_shape(* 
_output_shapes
:

ą
save/Assign_13AssignVariable_12/Adamsave/RestoreV2:13*
use_locking(*
T0*
_class
loc:@Variable_12*
validate_shape(* 
_output_shapes
:

ł
save/Assign_14AssignVariable_12/Adam_1save/RestoreV2:14*
use_locking(*
T0*
_class
loc:@Variable_12*
validate_shape(* 
_output_shapes
:

§
save/Assign_15AssignVariable_13save/RestoreV2:15*
use_locking(*
T0*
_class
loc:@Variable_13*
validate_shape(*
_output_shapes	
:
Ź
save/Assign_16AssignVariable_13/Adamsave/RestoreV2:16*
use_locking(*
T0*
_class
loc:@Variable_13*
validate_shape(*
_output_shapes	
:
Ž
save/Assign_17AssignVariable_13/Adam_1save/RestoreV2:17*
use_locking(*
T0*
_class
loc:@Variable_13*
validate_shape(*
_output_shapes	
:
Ź
save/Assign_18AssignVariable_14save/RestoreV2:18*
use_locking(*
T0*
_class
loc:@Variable_14*
validate_shape(* 
_output_shapes
:

ą
save/Assign_19AssignVariable_14/Adamsave/RestoreV2:19*
use_locking(*
T0*
_class
loc:@Variable_14*
validate_shape(* 
_output_shapes
:

ł
save/Assign_20AssignVariable_14/Adam_1save/RestoreV2:20*
use_locking(*
T0*
_class
loc:@Variable_14*
validate_shape(* 
_output_shapes
:

§
save/Assign_21AssignVariable_15save/RestoreV2:21*
use_locking(*
T0*
_class
loc:@Variable_15*
validate_shape(*
_output_shapes	
:
Ź
save/Assign_22AssignVariable_15/Adamsave/RestoreV2:22*
use_locking(*
T0*
_class
loc:@Variable_15*
validate_shape(*
_output_shapes	
:
Ž
save/Assign_23AssignVariable_15/Adam_1save/RestoreV2:23*
use_locking(*
T0*
_class
loc:@Variable_15*
validate_shape(*
_output_shapes	
:
Ť
save/Assign_24AssignVariable_16save/RestoreV2:24*
use_locking(*
T0*
_class
loc:@Variable_16*
validate_shape(*
_output_shapes
:	
°
save/Assign_25AssignVariable_16/Adamsave/RestoreV2:25*
use_locking(*
T0*
_class
loc:@Variable_16*
validate_shape(*
_output_shapes
:	
˛
save/Assign_26AssignVariable_16/Adam_1save/RestoreV2:26*
use_locking(*
T0*
_class
loc:@Variable_16*
validate_shape(*
_output_shapes
:	
Ś
save/Assign_27AssignVariable_17save/RestoreV2:27*
use_locking(*
T0*
_class
loc:@Variable_17*
validate_shape(*
_output_shapes
:
Ť
save/Assign_28AssignVariable_17/Adamsave/RestoreV2:28*
use_locking(*
T0*
_class
loc:@Variable_17*
validate_shape(*
_output_shapes
:
­
save/Assign_29AssignVariable_17/Adam_1save/RestoreV2:29*
use_locking(*
T0*
_class
loc:@Variable_17*
validate_shape(*
_output_shapes
:
Ş
save/Assign_30AssignVariable_18save/RestoreV2:30*
use_locking(*
T0*
_class
loc:@Variable_18*
validate_shape(*
_output_shapes

:
Ż
save/Assign_31AssignVariable_18/Adamsave/RestoreV2:31*
use_locking(*
T0*
_class
loc:@Variable_18*
validate_shape(*
_output_shapes

:
ą
save/Assign_32AssignVariable_18/Adam_1save/RestoreV2:32*
use_locking(*
T0*
_class
loc:@Variable_18*
validate_shape(*
_output_shapes

:
Ś
save/Assign_33AssignVariable_19save/RestoreV2:33*
use_locking(*
T0*
_class
loc:@Variable_19*
validate_shape(*
_output_shapes
:
Ť
save/Assign_34AssignVariable_19/Adamsave/RestoreV2:34*
use_locking(*
T0*
_class
loc:@Variable_19*
validate_shape(*
_output_shapes
:
­
save/Assign_35AssignVariable_19/Adam_1save/RestoreV2:35*
use_locking(*
T0*
_class
loc:@Variable_19*
validate_shape(*
_output_shapes
:
Ş
save/Assign_36Assign
Variable_2save/RestoreV2:36*
use_locking(*
T0*
_class
loc:@Variable_2*
validate_shape(* 
_output_shapes
:

Ż
save/Assign_37AssignVariable_2/Adamsave/RestoreV2:37*
use_locking(*
T0*
_class
loc:@Variable_2*
validate_shape(* 
_output_shapes
:

ą
save/Assign_38AssignVariable_2/Adam_1save/RestoreV2:38*
use_locking(*
T0*
_class
loc:@Variable_2*
validate_shape(* 
_output_shapes
:

Ľ
save/Assign_39Assign
Variable_3save/RestoreV2:39*
use_locking(*
T0*
_class
loc:@Variable_3*
validate_shape(*
_output_shapes	
:
Ş
save/Assign_40AssignVariable_3/Adamsave/RestoreV2:40*
use_locking(*
T0*
_class
loc:@Variable_3*
validate_shape(*
_output_shapes	
:
Ź
save/Assign_41AssignVariable_3/Adam_1save/RestoreV2:41*
use_locking(*
T0*
_class
loc:@Variable_3*
validate_shape(*
_output_shapes	
:
Ş
save/Assign_42Assign
Variable_4save/RestoreV2:42*
use_locking(*
T0*
_class
loc:@Variable_4*
validate_shape(* 
_output_shapes
:

Ż
save/Assign_43AssignVariable_4/Adamsave/RestoreV2:43*
use_locking(*
T0*
_class
loc:@Variable_4*
validate_shape(* 
_output_shapes
:

ą
save/Assign_44AssignVariable_4/Adam_1save/RestoreV2:44*
use_locking(*
T0*
_class
loc:@Variable_4*
validate_shape(* 
_output_shapes
:

Ľ
save/Assign_45Assign
Variable_5save/RestoreV2:45*
use_locking(*
T0*
_class
loc:@Variable_5*
validate_shape(*
_output_shapes	
:
Ş
save/Assign_46AssignVariable_5/Adamsave/RestoreV2:46*
use_locking(*
T0*
_class
loc:@Variable_5*
validate_shape(*
_output_shapes	
:
Ź
save/Assign_47AssignVariable_5/Adam_1save/RestoreV2:47*
use_locking(*
T0*
_class
loc:@Variable_5*
validate_shape(*
_output_shapes	
:
Ş
save/Assign_48Assign
Variable_6save/RestoreV2:48*
use_locking(*
T0*
_class
loc:@Variable_6*
validate_shape(* 
_output_shapes
:

Ż
save/Assign_49AssignVariable_6/Adamsave/RestoreV2:49*
use_locking(*
T0*
_class
loc:@Variable_6*
validate_shape(* 
_output_shapes
:

ą
save/Assign_50AssignVariable_6/Adam_1save/RestoreV2:50*
use_locking(*
T0*
_class
loc:@Variable_6*
validate_shape(* 
_output_shapes
:

Ľ
save/Assign_51Assign
Variable_7save/RestoreV2:51*
use_locking(*
T0*
_class
loc:@Variable_7*
validate_shape(*
_output_shapes	
:
Ş
save/Assign_52AssignVariable_7/Adamsave/RestoreV2:52*
use_locking(*
T0*
_class
loc:@Variable_7*
validate_shape(*
_output_shapes	
:
Ź
save/Assign_53AssignVariable_7/Adam_1save/RestoreV2:53*
use_locking(*
T0*
_class
loc:@Variable_7*
validate_shape(*
_output_shapes	
:
Ş
save/Assign_54Assign
Variable_8save/RestoreV2:54*
use_locking(*
T0*
_class
loc:@Variable_8*
validate_shape(* 
_output_shapes
:

Ż
save/Assign_55AssignVariable_8/Adamsave/RestoreV2:55*
use_locking(*
T0*
_class
loc:@Variable_8*
validate_shape(* 
_output_shapes
:

ą
save/Assign_56AssignVariable_8/Adam_1save/RestoreV2:56*
use_locking(*
T0*
_class
loc:@Variable_8*
validate_shape(* 
_output_shapes
:

Ľ
save/Assign_57Assign
Variable_9save/RestoreV2:57*
use_locking(*
T0*
_class
loc:@Variable_9*
validate_shape(*
_output_shapes	
:
Ş
save/Assign_58AssignVariable_9/Adamsave/RestoreV2:58*
use_locking(*
T0*
_class
loc:@Variable_9*
validate_shape(*
_output_shapes	
:
Ź
save/Assign_59AssignVariable_9/Adam_1save/RestoreV2:59*
use_locking(*
T0*
_class
loc:@Variable_9*
validate_shape(*
_output_shapes	
:

save/Assign_60Assignbeta1_powersave/RestoreV2:60*
use_locking(*
T0*
_class
loc:@Variable*
validate_shape(*
_output_shapes
: 

save/Assign_61Assignbeta2_powersave/RestoreV2:61*
use_locking(*
T0*
_class
loc:@Variable*
validate_shape(*
_output_shapes
: 
Ź
save/restore_shardNoOp^save/Assign^save/Assign_1^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19^save/Assign_2^save/Assign_20^save/Assign_21^save/Assign_22^save/Assign_23^save/Assign_24^save/Assign_25^save/Assign_26^save/Assign_27^save/Assign_28^save/Assign_29^save/Assign_3^save/Assign_30^save/Assign_31^save/Assign_32^save/Assign_33^save/Assign_34^save/Assign_35^save/Assign_36^save/Assign_37^save/Assign_38^save/Assign_39^save/Assign_4^save/Assign_40^save/Assign_41^save/Assign_42^save/Assign_43^save/Assign_44^save/Assign_45^save/Assign_46^save/Assign_47^save/Assign_48^save/Assign_49^save/Assign_5^save/Assign_50^save/Assign_51^save/Assign_52^save/Assign_53^save/Assign_54^save/Assign_55^save/Assign_56^save/Assign_57^save/Assign_58^save/Assign_59^save/Assign_6^save/Assign_60^save/Assign_61^save/Assign_7^save/Assign_8^save/Assign_9
-
save/restore_allNoOp^save/restore_shard "<
save/Const:0save/Identity:0save/restore_all (5 @F8"
trainable_variables˙ü
A

Variable:0Variable/AssignVariable/read:02random_normal:08
I
Variable_1:0Variable_1/AssignVariable_1/read:02random_normal_1:08
I
Variable_2:0Variable_2/AssignVariable_2/read:02random_normal_2:08
I
Variable_3:0Variable_3/AssignVariable_3/read:02random_normal_3:08
I
Variable_4:0Variable_4/AssignVariable_4/read:02random_normal_4:08
I
Variable_5:0Variable_5/AssignVariable_5/read:02random_normal_5:08
I
Variable_6:0Variable_6/AssignVariable_6/read:02random_normal_6:08
I
Variable_7:0Variable_7/AssignVariable_7/read:02random_normal_7:08
I
Variable_8:0Variable_8/AssignVariable_8/read:02random_normal_8:08
I
Variable_9:0Variable_9/AssignVariable_9/read:02random_normal_9:08
M
Variable_10:0Variable_10/AssignVariable_10/read:02random_normal_10:08
M
Variable_11:0Variable_11/AssignVariable_11/read:02random_normal_11:08
M
Variable_12:0Variable_12/AssignVariable_12/read:02random_normal_12:08
M
Variable_13:0Variable_13/AssignVariable_13/read:02random_normal_13:08
M
Variable_14:0Variable_14/AssignVariable_14/read:02random_normal_14:08
M
Variable_15:0Variable_15/AssignVariable_15/read:02random_normal_15:08
M
Variable_16:0Variable_16/AssignVariable_16/read:02random_normal_16:08
M
Variable_17:0Variable_17/AssignVariable_17/read:02random_normal_17:08
M
Variable_18:0Variable_18/AssignVariable_18/read:02random_normal_18:08
M
Variable_19:0Variable_19/AssignVariable_19/read:02random_normal_19:08"
train_op

Adam"Š0
	variables00
A

Variable:0Variable/AssignVariable/read:02random_normal:08
I
Variable_1:0Variable_1/AssignVariable_1/read:02random_normal_1:08
I
Variable_2:0Variable_2/AssignVariable_2/read:02random_normal_2:08
I
Variable_3:0Variable_3/AssignVariable_3/read:02random_normal_3:08
I
Variable_4:0Variable_4/AssignVariable_4/read:02random_normal_4:08
I
Variable_5:0Variable_5/AssignVariable_5/read:02random_normal_5:08
I
Variable_6:0Variable_6/AssignVariable_6/read:02random_normal_6:08
I
Variable_7:0Variable_7/AssignVariable_7/read:02random_normal_7:08
I
Variable_8:0Variable_8/AssignVariable_8/read:02random_normal_8:08
I
Variable_9:0Variable_9/AssignVariable_9/read:02random_normal_9:08
M
Variable_10:0Variable_10/AssignVariable_10/read:02random_normal_10:08
M
Variable_11:0Variable_11/AssignVariable_11/read:02random_normal_11:08
M
Variable_12:0Variable_12/AssignVariable_12/read:02random_normal_12:08
M
Variable_13:0Variable_13/AssignVariable_13/read:02random_normal_13:08
M
Variable_14:0Variable_14/AssignVariable_14/read:02random_normal_14:08
M
Variable_15:0Variable_15/AssignVariable_15/read:02random_normal_15:08
M
Variable_16:0Variable_16/AssignVariable_16/read:02random_normal_16:08
M
Variable_17:0Variable_17/AssignVariable_17/read:02random_normal_17:08
M
Variable_18:0Variable_18/AssignVariable_18/read:02random_normal_18:08
M
Variable_19:0Variable_19/AssignVariable_19/read:02random_normal_19:08
T
beta1_power:0beta1_power/Assignbeta1_power/read:02beta1_power/initial_value:0
T
beta2_power:0beta2_power/Assignbeta2_power/read:02beta2_power/initial_value:0
`
Variable/Adam:0Variable/Adam/AssignVariable/Adam/read:02!Variable/Adam/Initializer/zeros:0
h
Variable/Adam_1:0Variable/Adam_1/AssignVariable/Adam_1/read:02#Variable/Adam_1/Initializer/zeros:0
h
Variable_1/Adam:0Variable_1/Adam/AssignVariable_1/Adam/read:02#Variable_1/Adam/Initializer/zeros:0
p
Variable_1/Adam_1:0Variable_1/Adam_1/AssignVariable_1/Adam_1/read:02%Variable_1/Adam_1/Initializer/zeros:0
h
Variable_2/Adam:0Variable_2/Adam/AssignVariable_2/Adam/read:02#Variable_2/Adam/Initializer/zeros:0
p
Variable_2/Adam_1:0Variable_2/Adam_1/AssignVariable_2/Adam_1/read:02%Variable_2/Adam_1/Initializer/zeros:0
h
Variable_3/Adam:0Variable_3/Adam/AssignVariable_3/Adam/read:02#Variable_3/Adam/Initializer/zeros:0
p
Variable_3/Adam_1:0Variable_3/Adam_1/AssignVariable_3/Adam_1/read:02%Variable_3/Adam_1/Initializer/zeros:0
h
Variable_4/Adam:0Variable_4/Adam/AssignVariable_4/Adam/read:02#Variable_4/Adam/Initializer/zeros:0
p
Variable_4/Adam_1:0Variable_4/Adam_1/AssignVariable_4/Adam_1/read:02%Variable_4/Adam_1/Initializer/zeros:0
h
Variable_5/Adam:0Variable_5/Adam/AssignVariable_5/Adam/read:02#Variable_5/Adam/Initializer/zeros:0
p
Variable_5/Adam_1:0Variable_5/Adam_1/AssignVariable_5/Adam_1/read:02%Variable_5/Adam_1/Initializer/zeros:0
h
Variable_6/Adam:0Variable_6/Adam/AssignVariable_6/Adam/read:02#Variable_6/Adam/Initializer/zeros:0
p
Variable_6/Adam_1:0Variable_6/Adam_1/AssignVariable_6/Adam_1/read:02%Variable_6/Adam_1/Initializer/zeros:0
h
Variable_7/Adam:0Variable_7/Adam/AssignVariable_7/Adam/read:02#Variable_7/Adam/Initializer/zeros:0
p
Variable_7/Adam_1:0Variable_7/Adam_1/AssignVariable_7/Adam_1/read:02%Variable_7/Adam_1/Initializer/zeros:0
h
Variable_8/Adam:0Variable_8/Adam/AssignVariable_8/Adam/read:02#Variable_8/Adam/Initializer/zeros:0
p
Variable_8/Adam_1:0Variable_8/Adam_1/AssignVariable_8/Adam_1/read:02%Variable_8/Adam_1/Initializer/zeros:0
h
Variable_9/Adam:0Variable_9/Adam/AssignVariable_9/Adam/read:02#Variable_9/Adam/Initializer/zeros:0
p
Variable_9/Adam_1:0Variable_9/Adam_1/AssignVariable_9/Adam_1/read:02%Variable_9/Adam_1/Initializer/zeros:0
l
Variable_10/Adam:0Variable_10/Adam/AssignVariable_10/Adam/read:02$Variable_10/Adam/Initializer/zeros:0
t
Variable_10/Adam_1:0Variable_10/Adam_1/AssignVariable_10/Adam_1/read:02&Variable_10/Adam_1/Initializer/zeros:0
l
Variable_11/Adam:0Variable_11/Adam/AssignVariable_11/Adam/read:02$Variable_11/Adam/Initializer/zeros:0
t
Variable_11/Adam_1:0Variable_11/Adam_1/AssignVariable_11/Adam_1/read:02&Variable_11/Adam_1/Initializer/zeros:0
l
Variable_12/Adam:0Variable_12/Adam/AssignVariable_12/Adam/read:02$Variable_12/Adam/Initializer/zeros:0
t
Variable_12/Adam_1:0Variable_12/Adam_1/AssignVariable_12/Adam_1/read:02&Variable_12/Adam_1/Initializer/zeros:0
l
Variable_13/Adam:0Variable_13/Adam/AssignVariable_13/Adam/read:02$Variable_13/Adam/Initializer/zeros:0
t
Variable_13/Adam_1:0Variable_13/Adam_1/AssignVariable_13/Adam_1/read:02&Variable_13/Adam_1/Initializer/zeros:0
l
Variable_14/Adam:0Variable_14/Adam/AssignVariable_14/Adam/read:02$Variable_14/Adam/Initializer/zeros:0
t
Variable_14/Adam_1:0Variable_14/Adam_1/AssignVariable_14/Adam_1/read:02&Variable_14/Adam_1/Initializer/zeros:0
l
Variable_15/Adam:0Variable_15/Adam/AssignVariable_15/Adam/read:02$Variable_15/Adam/Initializer/zeros:0
t
Variable_15/Adam_1:0Variable_15/Adam_1/AssignVariable_15/Adam_1/read:02&Variable_15/Adam_1/Initializer/zeros:0
l
Variable_16/Adam:0Variable_16/Adam/AssignVariable_16/Adam/read:02$Variable_16/Adam/Initializer/zeros:0
t
Variable_16/Adam_1:0Variable_16/Adam_1/AssignVariable_16/Adam_1/read:02&Variable_16/Adam_1/Initializer/zeros:0
l
Variable_17/Adam:0Variable_17/Adam/AssignVariable_17/Adam/read:02$Variable_17/Adam/Initializer/zeros:0
t
Variable_17/Adam_1:0Variable_17/Adam_1/AssignVariable_17/Adam_1/read:02&Variable_17/Adam_1/Initializer/zeros:0
l
Variable_18/Adam:0Variable_18/Adam/AssignVariable_18/Adam/read:02$Variable_18/Adam/Initializer/zeros:0
t
Variable_18/Adam_1:0Variable_18/Adam_1/AssignVariable_18/Adam_1/read:02&Variable_18/Adam_1/Initializer/zeros:0
l
Variable_19/Adam:0Variable_19/Adam/AssignVariable_19/Adam/read:02$Variable_19/Adam/Initializer/zeros:0
t
Variable_19/Adam_1:0Variable_19/Adam_1/AssignVariable_19/Adam_1/read:02&Variable_19/Adam_1/Initializer/zeros:0*
serving_defaultu
+
myInput 
	myInput:0˙˙˙˙˙˙˙˙˙*
myOutput
add_9:0˙˙˙˙˙˙˙˙˙tensorflow/serving/predict