       ЃK"	  РеыЃжAbrain.Event:2=Dўх1     Въ	!ЊбеыЃжA"Є
p
dense_1_inputPlaceholder*
dtype0*
shape:џџџџџџџџџ*'
_output_shapes
:џџџџџџџџџ
m
dense_1/random_uniform/shapeConst*
valueB"   @   *
dtype0*
_output_shapes
:
_
dense_1/random_uniform/minConst*
valueB
 *!ьО*
dtype0*
_output_shapes
: 
_
dense_1/random_uniform/maxConst*
valueB
 *!ь>*
dtype0*
_output_shapes
: 
Ј
$dense_1/random_uniform/RandomUniformRandomUniformdense_1/random_uniform/shape*
seedБџх)*
seed2Дї*
dtype0*
T0*
_output_shapes

:@
z
dense_1/random_uniform/subSubdense_1/random_uniform/maxdense_1/random_uniform/min*
T0*
_output_shapes
: 

dense_1/random_uniform/mulMul$dense_1/random_uniform/RandomUniformdense_1/random_uniform/sub*
T0*
_output_shapes

:@
~
dense_1/random_uniformAdddense_1/random_uniform/muldense_1/random_uniform/min*
T0*
_output_shapes

:@

dense_1/kernel
VariableV2*
shape
:@*
dtype0*
	container *
shared_name *
_output_shapes

:@
М
dense_1/kernel/AssignAssigndense_1/kerneldense_1/random_uniform*
T0*
validate_shape(*
use_locking(*!
_class
loc:@dense_1/kernel*
_output_shapes

:@
{
dense_1/kernel/readIdentitydense_1/kernel*
T0*!
_class
loc:@dense_1/kernel*
_output_shapes

:@
Z
dense_1/ConstConst*
valueB@*    *
dtype0*
_output_shapes
:@
x
dense_1/bias
VariableV2*
shape:@*
dtype0*
	container *
shared_name *
_output_shapes
:@
Љ
dense_1/bias/AssignAssigndense_1/biasdense_1/Const*
T0*
validate_shape(*
use_locking(*
_class
loc:@dense_1/bias*
_output_shapes
:@
q
dense_1/bias/readIdentitydense_1/bias*
T0*
_class
loc:@dense_1/bias*
_output_shapes
:@

dense_1/MatMulMatMuldense_1_inputdense_1/kernel/read*
transpose_a( *
transpose_b( *
T0*'
_output_shapes
:џџџџџџџџџ@

dense_1/BiasAddBiasAdddense_1/MatMuldense_1/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:џџџџџџџџџ@
W
dense_1/ReluReludense_1/BiasAdd*
T0*'
_output_shapes
:џџџџџџџџџ@
m
dense_2/random_uniform/shapeConst*
valueB"@   @   *
dtype0*
_output_shapes
:
_
dense_2/random_uniform/minConst*
valueB
 *зГ]О*
dtype0*
_output_shapes
: 
_
dense_2/random_uniform/maxConst*
valueB
 *зГ]>*
dtype0*
_output_shapes
: 
Ј
$dense_2/random_uniform/RandomUniformRandomUniformdense_2/random_uniform/shape*
seedБџх)*
seed2џЙѕ*
dtype0*
T0*
_output_shapes

:@@
z
dense_2/random_uniform/subSubdense_2/random_uniform/maxdense_2/random_uniform/min*
T0*
_output_shapes
: 

dense_2/random_uniform/mulMul$dense_2/random_uniform/RandomUniformdense_2/random_uniform/sub*
T0*
_output_shapes

:@@
~
dense_2/random_uniformAdddense_2/random_uniform/muldense_2/random_uniform/min*
T0*
_output_shapes

:@@

dense_2/kernel
VariableV2*
shape
:@@*
dtype0*
	container *
shared_name *
_output_shapes

:@@
М
dense_2/kernel/AssignAssigndense_2/kerneldense_2/random_uniform*
T0*
validate_shape(*
use_locking(*!
_class
loc:@dense_2/kernel*
_output_shapes

:@@
{
dense_2/kernel/readIdentitydense_2/kernel*
T0*!
_class
loc:@dense_2/kernel*
_output_shapes

:@@
Z
dense_2/ConstConst*
valueB@*    *
dtype0*
_output_shapes
:@
x
dense_2/bias
VariableV2*
shape:@*
dtype0*
	container *
shared_name *
_output_shapes
:@
Љ
dense_2/bias/AssignAssigndense_2/biasdense_2/Const*
T0*
validate_shape(*
use_locking(*
_class
loc:@dense_2/bias*
_output_shapes
:@
q
dense_2/bias/readIdentitydense_2/bias*
T0*
_class
loc:@dense_2/bias*
_output_shapes
:@

dense_2/MatMulMatMuldense_1/Reludense_2/kernel/read*
transpose_a( *
transpose_b( *
T0*'
_output_shapes
:џџџџџџџџџ@

dense_2/BiasAddBiasAdddense_2/MatMuldense_2/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:џџџџџџџџџ@
W
dense_2/ReluReludense_2/BiasAdd*
T0*'
_output_shapes
:џџџџџџџџџ@
m
dense_3/random_uniform/shapeConst*
valueB"@      *
dtype0*
_output_shapes
:
_
dense_3/random_uniform/minConst*
valueB
 *О*
dtype0*
_output_shapes
: 
_
dense_3/random_uniform/maxConst*
valueB
 *>*
dtype0*
_output_shapes
: 
Ј
$dense_3/random_uniform/RandomUniformRandomUniformdense_3/random_uniform/shape*
seedБџх)*
seed2ші*
dtype0*
T0*
_output_shapes

:@
z
dense_3/random_uniform/subSubdense_3/random_uniform/maxdense_3/random_uniform/min*
T0*
_output_shapes
: 

dense_3/random_uniform/mulMul$dense_3/random_uniform/RandomUniformdense_3/random_uniform/sub*
T0*
_output_shapes

:@
~
dense_3/random_uniformAdddense_3/random_uniform/muldense_3/random_uniform/min*
T0*
_output_shapes

:@

dense_3/kernel
VariableV2*
shape
:@*
dtype0*
	container *
shared_name *
_output_shapes

:@
М
dense_3/kernel/AssignAssigndense_3/kerneldense_3/random_uniform*
T0*
validate_shape(*
use_locking(*!
_class
loc:@dense_3/kernel*
_output_shapes

:@
{
dense_3/kernel/readIdentitydense_3/kernel*
T0*!
_class
loc:@dense_3/kernel*
_output_shapes

:@
Z
dense_3/ConstConst*
valueB*    *
dtype0*
_output_shapes
:
x
dense_3/bias
VariableV2*
shape:*
dtype0*
	container *
shared_name *
_output_shapes
:
Љ
dense_3/bias/AssignAssigndense_3/biasdense_3/Const*
T0*
validate_shape(*
use_locking(*
_class
loc:@dense_3/bias*
_output_shapes
:
q
dense_3/bias/readIdentitydense_3/bias*
T0*
_class
loc:@dense_3/bias*
_output_shapes
:

dense_3/MatMulMatMuldense_2/Reludense_3/kernel/read*
transpose_a( *
transpose_b( *
T0*'
_output_shapes
:џџџџџџџџџ

dense_3/BiasAddBiasAdddense_3/MatMuldense_3/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:џџџџџџџџџ
]
RMSprop/lr/initial_valueConst*
valueB
 *o:*
dtype0*
_output_shapes
: 
n

RMSprop/lr
VariableV2*
shape: *
dtype0*
	container *
shared_name *
_output_shapes
: 
Њ
RMSprop/lr/AssignAssign
RMSprop/lrRMSprop/lr/initial_value*
T0*
validate_shape(*
use_locking(*
_class
loc:@RMSprop/lr*
_output_shapes
: 
g
RMSprop/lr/readIdentity
RMSprop/lr*
T0*
_class
loc:@RMSprop/lr*
_output_shapes
: 
^
RMSprop/rho/initial_valueConst*
valueB
 *fff?*
dtype0*
_output_shapes
: 
o
RMSprop/rho
VariableV2*
shape: *
dtype0*
	container *
shared_name *
_output_shapes
: 
Ў
RMSprop/rho/AssignAssignRMSprop/rhoRMSprop/rho/initial_value*
T0*
validate_shape(*
use_locking(*
_class
loc:@RMSprop/rho*
_output_shapes
: 
j
RMSprop/rho/readIdentityRMSprop/rho*
T0*
_class
loc:@RMSprop/rho*
_output_shapes
: 
`
RMSprop/decay/initial_valueConst*
valueB
 *    *
dtype0*
_output_shapes
: 
q
RMSprop/decay
VariableV2*
shape: *
dtype0*
	container *
shared_name *
_output_shapes
: 
Ж
RMSprop/decay/AssignAssignRMSprop/decayRMSprop/decay/initial_value*
T0*
validate_shape(*
use_locking(* 
_class
loc:@RMSprop/decay*
_output_shapes
: 
p
RMSprop/decay/readIdentityRMSprop/decay*
T0* 
_class
loc:@RMSprop/decay*
_output_shapes
: 
b
 RMSprop/iterations/initial_valueConst*
value	B	 R *
dtype0	*
_output_shapes
: 
v
RMSprop/iterations
VariableV2*
shape: *
dtype0	*
	container *
shared_name *
_output_shapes
: 
Ъ
RMSprop/iterations/AssignAssignRMSprop/iterations RMSprop/iterations/initial_value*
T0	*
validate_shape(*
use_locking(*%
_class
loc:@RMSprop/iterations*
_output_shapes
: 

RMSprop/iterations/readIdentityRMSprop/iterations*
T0	*%
_class
loc:@RMSprop/iterations*
_output_shapes
: 

dense_3_targetPlaceholder*
dtype0*%
shape:џџџџџџџџџџџџџџџџџџ*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
q
dense_3_sample_weightsPlaceholder*
dtype0*
shape:џџџџџџџџџ*#
_output_shapes
:џџџџџџџџџ
x
loss/dense_3_loss/subSubdense_3/BiasAdddense_3_target*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
t
loss/dense_3_loss/SquareSquareloss/dense_3_loss/sub*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
s
(loss/dense_3_loss/Mean/reduction_indicesConst*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 
­
loss/dense_3_loss/MeanMeanloss/dense_3_loss/Square(loss/dense_3_loss/Mean/reduction_indices*
	keep_dims( *
T0*

Tidx0*#
_output_shapes
:џџџџџџџџџ
m
*loss/dense_3_loss/Mean_1/reduction_indicesConst*
valueB *
dtype0*
_output_shapes
: 
Џ
loss/dense_3_loss/Mean_1Meanloss/dense_3_loss/Mean*loss/dense_3_loss/Mean_1/reduction_indices*
	keep_dims( *
T0*

Tidx0*#
_output_shapes
:џџџџџџџџџ
|
loss/dense_3_loss/mulMulloss/dense_3_loss/Mean_1dense_3_sample_weights*
T0*#
_output_shapes
:џџџџџџџџџ
a
loss/dense_3_loss/NotEqual/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 

loss/dense_3_loss/NotEqualNotEqualdense_3_sample_weightsloss/dense_3_loss/NotEqual/y*
T0*#
_output_shapes
:џџџџџџџџџ
w
loss/dense_3_loss/CastCastloss/dense_3_loss/NotEqual*

SrcT0
*

DstT0*#
_output_shapes
:џџџџџџџџџ
a
loss/dense_3_loss/ConstConst*
valueB: *
dtype0*
_output_shapes
:

loss/dense_3_loss/Mean_2Meanloss/dense_3_loss/Castloss/dense_3_loss/Const*
	keep_dims( *
T0*

Tidx0*
_output_shapes
: 

loss/dense_3_loss/truedivRealDivloss/dense_3_loss/mulloss/dense_3_loss/Mean_2*
T0*#
_output_shapes
:џџџџџџџџџ
c
loss/dense_3_loss/Const_1Const*
valueB: *
dtype0*
_output_shapes
:

loss/dense_3_loss/Mean_3Meanloss/dense_3_loss/truedivloss/dense_3_loss/Const_1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
: 
O

loss/mul/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
V
loss/mulMul
loss/mul/xloss/dense_3_loss/Mean_3*
T0*
_output_shapes
: 

metrics/mean_absolute_error/subSubdense_3/BiasAdddense_3_target*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ

metrics/mean_absolute_error/AbsAbsmetrics/mean_absolute_error/sub*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
}
2metrics/mean_absolute_error/Mean/reduction_indicesConst*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 
Ш
 metrics/mean_absolute_error/MeanMeanmetrics/mean_absolute_error/Abs2metrics/mean_absolute_error/Mean/reduction_indices*
	keep_dims( *
T0*

Tidx0*#
_output_shapes
:џџџџџџџџџ
k
!metrics/mean_absolute_error/ConstConst*
valueB: *
dtype0*
_output_shapes
:
­
"metrics/mean_absolute_error/Mean_1Mean metrics/mean_absolute_error/Mean!metrics/mean_absolute_error/Const*
	keep_dims( *
T0*

Tidx0*
_output_shapes
: 

 training/RMSprop/gradients/ShapeConst*
valueB *
dtype0*
_class
loc:@loss/mul*
_output_shapes
: 

$training/RMSprop/gradients/grad_ys_0Const*
valueB
 *  ?*
dtype0*
_class
loc:@loss/mul*
_output_shapes
: 
­
training/RMSprop/gradients/FillFill training/RMSprop/gradients/Shape$training/RMSprop/gradients/grad_ys_0*
T0*
_class
loc:@loss/mul*
_output_shapes
: 
Ќ
,training/RMSprop/gradients/loss/mul_grad/MulMultraining/RMSprop/gradients/Fillloss/dense_3_loss/Mean_3*
T0*
_class
loc:@loss/mul*
_output_shapes
: 
 
.training/RMSprop/gradients/loss/mul_grad/Mul_1Multraining/RMSprop/gradients/Fill
loss/mul/x*
T0*
_class
loc:@loss/mul*
_output_shapes
: 
Н
Ftraining/RMSprop/gradients/loss/dense_3_loss/Mean_3_grad/Reshape/shapeConst*
valueB:*
dtype0*+
_class!
loc:@loss/dense_3_loss/Mean_3*
_output_shapes
:
Ѓ
@training/RMSprop/gradients/loss/dense_3_loss/Mean_3_grad/ReshapeReshape.training/RMSprop/gradients/loss/mul_grad/Mul_1Ftraining/RMSprop/gradients/loss/dense_3_loss/Mean_3_grad/Reshape/shape*
T0*
Tshape0*+
_class!
loc:@loss/dense_3_loss/Mean_3*
_output_shapes
:
Ф
>training/RMSprop/gradients/loss/dense_3_loss/Mean_3_grad/ShapeShapeloss/dense_3_loss/truediv*
T0*
out_type0*+
_class!
loc:@loss/dense_3_loss/Mean_3*
_output_shapes
:
Д
=training/RMSprop/gradients/loss/dense_3_loss/Mean_3_grad/TileTile@training/RMSprop/gradients/loss/dense_3_loss/Mean_3_grad/Reshape>training/RMSprop/gradients/loss/dense_3_loss/Mean_3_grad/Shape*
T0*

Tmultiples0*+
_class!
loc:@loss/dense_3_loss/Mean_3*#
_output_shapes
:џџџџџџџџџ
Ц
@training/RMSprop/gradients/loss/dense_3_loss/Mean_3_grad/Shape_1Shapeloss/dense_3_loss/truediv*
T0*
out_type0*+
_class!
loc:@loss/dense_3_loss/Mean_3*
_output_shapes
:
А
@training/RMSprop/gradients/loss/dense_3_loss/Mean_3_grad/Shape_2Const*
valueB *
dtype0*+
_class!
loc:@loss/dense_3_loss/Mean_3*
_output_shapes
: 
Е
>training/RMSprop/gradients/loss/dense_3_loss/Mean_3_grad/ConstConst*
valueB: *
dtype0*+
_class!
loc:@loss/dense_3_loss/Mean_3*
_output_shapes
:
В
=training/RMSprop/gradients/loss/dense_3_loss/Mean_3_grad/ProdProd@training/RMSprop/gradients/loss/dense_3_loss/Mean_3_grad/Shape_1>training/RMSprop/gradients/loss/dense_3_loss/Mean_3_grad/Const*
	keep_dims( *
T0*

Tidx0*+
_class!
loc:@loss/dense_3_loss/Mean_3*
_output_shapes
: 
З
@training/RMSprop/gradients/loss/dense_3_loss/Mean_3_grad/Const_1Const*
valueB: *
dtype0*+
_class!
loc:@loss/dense_3_loss/Mean_3*
_output_shapes
:
Ж
?training/RMSprop/gradients/loss/dense_3_loss/Mean_3_grad/Prod_1Prod@training/RMSprop/gradients/loss/dense_3_loss/Mean_3_grad/Shape_2@training/RMSprop/gradients/loss/dense_3_loss/Mean_3_grad/Const_1*
	keep_dims( *
T0*

Tidx0*+
_class!
loc:@loss/dense_3_loss/Mean_3*
_output_shapes
: 
Б
Btraining/RMSprop/gradients/loss/dense_3_loss/Mean_3_grad/Maximum/yConst*
value	B :*
dtype0*+
_class!
loc:@loss/dense_3_loss/Mean_3*
_output_shapes
: 

@training/RMSprop/gradients/loss/dense_3_loss/Mean_3_grad/MaximumMaximum?training/RMSprop/gradients/loss/dense_3_loss/Mean_3_grad/Prod_1Btraining/RMSprop/gradients/loss/dense_3_loss/Mean_3_grad/Maximum/y*
T0*+
_class!
loc:@loss/dense_3_loss/Mean_3*
_output_shapes
: 

Atraining/RMSprop/gradients/loss/dense_3_loss/Mean_3_grad/floordivFloorDiv=training/RMSprop/gradients/loss/dense_3_loss/Mean_3_grad/Prod@training/RMSprop/gradients/loss/dense_3_loss/Mean_3_grad/Maximum*
T0*+
_class!
loc:@loss/dense_3_loss/Mean_3*
_output_shapes
: 
х
=training/RMSprop/gradients/loss/dense_3_loss/Mean_3_grad/CastCastAtraining/RMSprop/gradients/loss/dense_3_loss/Mean_3_grad/floordiv*

SrcT0*

DstT0*+
_class!
loc:@loss/dense_3_loss/Mean_3*
_output_shapes
: 
Є
@training/RMSprop/gradients/loss/dense_3_loss/Mean_3_grad/truedivRealDiv=training/RMSprop/gradients/loss/dense_3_loss/Mean_3_grad/Tile=training/RMSprop/gradients/loss/dense_3_loss/Mean_3_grad/Cast*
T0*+
_class!
loc:@loss/dense_3_loss/Mean_3*#
_output_shapes
:џџџџџџџџџ
Т
?training/RMSprop/gradients/loss/dense_3_loss/truediv_grad/ShapeShapeloss/dense_3_loss/mul*
T0*
out_type0*,
_class"
 loc:@loss/dense_3_loss/truediv*
_output_shapes
:
В
Atraining/RMSprop/gradients/loss/dense_3_loss/truediv_grad/Shape_1Const*
valueB *
dtype0*,
_class"
 loc:@loss/dense_3_loss/truediv*
_output_shapes
: 
з
Otraining/RMSprop/gradients/loss/dense_3_loss/truediv_grad/BroadcastGradientArgsBroadcastGradientArgs?training/RMSprop/gradients/loss/dense_3_loss/truediv_grad/ShapeAtraining/RMSprop/gradients/loss/dense_3_loss/truediv_grad/Shape_1*
T0*,
_class"
 loc:@loss/dense_3_loss/truediv*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ

Atraining/RMSprop/gradients/loss/dense_3_loss/truediv_grad/RealDivRealDiv@training/RMSprop/gradients/loss/dense_3_loss/Mean_3_grad/truedivloss/dense_3_loss/Mean_2*
T0*,
_class"
 loc:@loss/dense_3_loss/truediv*#
_output_shapes
:џџџџџџџџџ
Ц
=training/RMSprop/gradients/loss/dense_3_loss/truediv_grad/SumSumAtraining/RMSprop/gradients/loss/dense_3_loss/truediv_grad/RealDivOtraining/RMSprop/gradients/loss/dense_3_loss/truediv_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*,
_class"
 loc:@loss/dense_3_loss/truediv*
_output_shapes
:
Ж
Atraining/RMSprop/gradients/loss/dense_3_loss/truediv_grad/ReshapeReshape=training/RMSprop/gradients/loss/dense_3_loss/truediv_grad/Sum?training/RMSprop/gradients/loss/dense_3_loss/truediv_grad/Shape*
T0*
Tshape0*,
_class"
 loc:@loss/dense_3_loss/truediv*#
_output_shapes
:џџџџџџџџџ
З
=training/RMSprop/gradients/loss/dense_3_loss/truediv_grad/NegNegloss/dense_3_loss/mul*
T0*,
_class"
 loc:@loss/dense_3_loss/truediv*#
_output_shapes
:џџџџџџџџџ

Ctraining/RMSprop/gradients/loss/dense_3_loss/truediv_grad/RealDiv_1RealDiv=training/RMSprop/gradients/loss/dense_3_loss/truediv_grad/Negloss/dense_3_loss/Mean_2*
T0*,
_class"
 loc:@loss/dense_3_loss/truediv*#
_output_shapes
:џџџџџџџџџ

Ctraining/RMSprop/gradients/loss/dense_3_loss/truediv_grad/RealDiv_2RealDivCtraining/RMSprop/gradients/loss/dense_3_loss/truediv_grad/RealDiv_1loss/dense_3_loss/Mean_2*
T0*,
_class"
 loc:@loss/dense_3_loss/truediv*#
_output_shapes
:џџџџџџџџџ
Ї
=training/RMSprop/gradients/loss/dense_3_loss/truediv_grad/mulMul@training/RMSprop/gradients/loss/dense_3_loss/Mean_3_grad/truedivCtraining/RMSprop/gradients/loss/dense_3_loss/truediv_grad/RealDiv_2*
T0*,
_class"
 loc:@loss/dense_3_loss/truediv*#
_output_shapes
:џџџџџџџџџ
Ц
?training/RMSprop/gradients/loss/dense_3_loss/truediv_grad/Sum_1Sum=training/RMSprop/gradients/loss/dense_3_loss/truediv_grad/mulQtraining/RMSprop/gradients/loss/dense_3_loss/truediv_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*,
_class"
 loc:@loss/dense_3_loss/truediv*
_output_shapes
:
Џ
Ctraining/RMSprop/gradients/loss/dense_3_loss/truediv_grad/Reshape_1Reshape?training/RMSprop/gradients/loss/dense_3_loss/truediv_grad/Sum_1Atraining/RMSprop/gradients/loss/dense_3_loss/truediv_grad/Shape_1*
T0*
Tshape0*,
_class"
 loc:@loss/dense_3_loss/truediv*
_output_shapes
: 
Н
;training/RMSprop/gradients/loss/dense_3_loss/mul_grad/ShapeShapeloss/dense_3_loss/Mean_1*
T0*
out_type0*(
_class
loc:@loss/dense_3_loss/mul*
_output_shapes
:
Н
=training/RMSprop/gradients/loss/dense_3_loss/mul_grad/Shape_1Shapedense_3_sample_weights*
T0*
out_type0*(
_class
loc:@loss/dense_3_loss/mul*
_output_shapes
:
Ч
Ktraining/RMSprop/gradients/loss/dense_3_loss/mul_grad/BroadcastGradientArgsBroadcastGradientArgs;training/RMSprop/gradients/loss/dense_3_loss/mul_grad/Shape=training/RMSprop/gradients/loss/dense_3_loss/mul_grad/Shape_1*
T0*(
_class
loc:@loss/dense_3_loss/mul*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
ѓ
9training/RMSprop/gradients/loss/dense_3_loss/mul_grad/mulMulAtraining/RMSprop/gradients/loss/dense_3_loss/truediv_grad/Reshapedense_3_sample_weights*
T0*(
_class
loc:@loss/dense_3_loss/mul*#
_output_shapes
:џџџџџџџџџ
В
9training/RMSprop/gradients/loss/dense_3_loss/mul_grad/SumSum9training/RMSprop/gradients/loss/dense_3_loss/mul_grad/mulKtraining/RMSprop/gradients/loss/dense_3_loss/mul_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*(
_class
loc:@loss/dense_3_loss/mul*
_output_shapes
:
І
=training/RMSprop/gradients/loss/dense_3_loss/mul_grad/ReshapeReshape9training/RMSprop/gradients/loss/dense_3_loss/mul_grad/Sum;training/RMSprop/gradients/loss/dense_3_loss/mul_grad/Shape*
T0*
Tshape0*(
_class
loc:@loss/dense_3_loss/mul*#
_output_shapes
:џџџџџџџџџ
ї
;training/RMSprop/gradients/loss/dense_3_loss/mul_grad/mul_1Mulloss/dense_3_loss/Mean_1Atraining/RMSprop/gradients/loss/dense_3_loss/truediv_grad/Reshape*
T0*(
_class
loc:@loss/dense_3_loss/mul*#
_output_shapes
:џџџџџџџџџ
И
;training/RMSprop/gradients/loss/dense_3_loss/mul_grad/Sum_1Sum;training/RMSprop/gradients/loss/dense_3_loss/mul_grad/mul_1Mtraining/RMSprop/gradients/loss/dense_3_loss/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*(
_class
loc:@loss/dense_3_loss/mul*
_output_shapes
:
Ќ
?training/RMSprop/gradients/loss/dense_3_loss/mul_grad/Reshape_1Reshape;training/RMSprop/gradients/loss/dense_3_loss/mul_grad/Sum_1=training/RMSprop/gradients/loss/dense_3_loss/mul_grad/Shape_1*
T0*
Tshape0*(
_class
loc:@loss/dense_3_loss/mul*#
_output_shapes
:џџџџџџџџџ
С
>training/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/ShapeShapeloss/dense_3_loss/Mean*
T0*
out_type0*+
_class!
loc:@loss/dense_3_loss/Mean_1*
_output_shapes
:
Ќ
=training/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/SizeConst*
value	B :*
dtype0*+
_class!
loc:@loss/dense_3_loss/Mean_1*
_output_shapes
: 
ў
<training/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/addAdd*loss/dense_3_loss/Mean_1/reduction_indices=training/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/Size*
T0*+
_class!
loc:@loss/dense_3_loss/Mean_1*
_output_shapes
: 

<training/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/modFloorMod<training/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/add=training/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/Size*
T0*+
_class!
loc:@loss/dense_3_loss/Mean_1*
_output_shapes
: 
З
@training/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/Shape_1Const*
valueB: *
dtype0*+
_class!
loc:@loss/dense_3_loss/Mean_1*
_output_shapes
:
Г
Dtraining/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/range/startConst*
value	B : *
dtype0*+
_class!
loc:@loss/dense_3_loss/Mean_1*
_output_shapes
: 
Г
Dtraining/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/range/deltaConst*
value	B :*
dtype0*+
_class!
loc:@loss/dense_3_loss/Mean_1*
_output_shapes
: 
ч
>training/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/rangeRangeDtraining/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/range/start=training/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/SizeDtraining/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/range/delta*

Tidx0*+
_class!
loc:@loss/dense_3_loss/Mean_1*
_output_shapes
:
В
Ctraining/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/Fill/valueConst*
value	B :*
dtype0*+
_class!
loc:@loss/dense_3_loss/Mean_1*
_output_shapes
: 

=training/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/FillFill@training/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/Shape_1Ctraining/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/Fill/value*
T0*+
_class!
loc:@loss/dense_3_loss/Mean_1*
_output_shapes
: 
И
Ftraining/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/DynamicStitchDynamicStitch>training/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/range<training/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/mod>training/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/Shape=training/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/Fill*
N*
T0*+
_class!
loc:@loss/dense_3_loss/Mean_1*#
_output_shapes
:џџџџџџџџџ
Б
Btraining/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/Maximum/yConst*
value	B :*
dtype0*+
_class!
loc:@loss/dense_3_loss/Mean_1*
_output_shapes
: 
В
@training/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/MaximumMaximumFtraining/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/DynamicStitchBtraining/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/Maximum/y*
T0*+
_class!
loc:@loss/dense_3_loss/Mean_1*#
_output_shapes
:џџџџџџџџџ
Њ
Atraining/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/floordivFloorDiv>training/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/Shape@training/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/Maximum*
T0*+
_class!
loc:@loss/dense_3_loss/Mean_1*#
_output_shapes
:џџџџџџџџџ
А
@training/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/ReshapeReshape=training/RMSprop/gradients/loss/dense_3_loss/mul_grad/ReshapeFtraining/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/DynamicStitch*
T0*
Tshape0*+
_class!
loc:@loss/dense_3_loss/Mean_1*
_output_shapes
:
Ќ
=training/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/TileTile@training/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/ReshapeAtraining/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/floordiv*
T0*

Tmultiples0*+
_class!
loc:@loss/dense_3_loss/Mean_1*
_output_shapes
:
У
@training/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/Shape_2Shapeloss/dense_3_loss/Mean*
T0*
out_type0*+
_class!
loc:@loss/dense_3_loss/Mean_1*
_output_shapes
:
Х
@training/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/Shape_3Shapeloss/dense_3_loss/Mean_1*
T0*
out_type0*+
_class!
loc:@loss/dense_3_loss/Mean_1*
_output_shapes
:
Е
>training/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/ConstConst*
valueB: *
dtype0*+
_class!
loc:@loss/dense_3_loss/Mean_1*
_output_shapes
:
В
=training/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/ProdProd@training/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/Shape_2>training/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/Const*
	keep_dims( *
T0*

Tidx0*+
_class!
loc:@loss/dense_3_loss/Mean_1*
_output_shapes
: 
З
@training/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/Const_1Const*
valueB: *
dtype0*+
_class!
loc:@loss/dense_3_loss/Mean_1*
_output_shapes
:
Ж
?training/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/Prod_1Prod@training/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/Shape_3@training/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/Const_1*
	keep_dims( *
T0*

Tidx0*+
_class!
loc:@loss/dense_3_loss/Mean_1*
_output_shapes
: 
Г
Dtraining/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/Maximum_1/yConst*
value	B :*
dtype0*+
_class!
loc:@loss/dense_3_loss/Mean_1*
_output_shapes
: 
Ђ
Btraining/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/Maximum_1Maximum?training/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/Prod_1Dtraining/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/Maximum_1/y*
T0*+
_class!
loc:@loss/dense_3_loss/Mean_1*
_output_shapes
: 
 
Ctraining/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/floordiv_1FloorDiv=training/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/ProdBtraining/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/Maximum_1*
T0*+
_class!
loc:@loss/dense_3_loss/Mean_1*
_output_shapes
: 
ч
=training/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/CastCastCtraining/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/floordiv_1*

SrcT0*

DstT0*+
_class!
loc:@loss/dense_3_loss/Mean_1*
_output_shapes
: 
Є
@training/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/truedivRealDiv=training/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/Tile=training/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/Cast*
T0*+
_class!
loc:@loss/dense_3_loss/Mean_1*#
_output_shapes
:џџџџџџџџџ
П
<training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/ShapeShapeloss/dense_3_loss/Square*
T0*
out_type0*)
_class
loc:@loss/dense_3_loss/Mean*
_output_shapes
:
Ј
;training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/SizeConst*
value	B :*
dtype0*)
_class
loc:@loss/dense_3_loss/Mean*
_output_shapes
: 
є
:training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/addAdd(loss/dense_3_loss/Mean/reduction_indices;training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/Size*
T0*)
_class
loc:@loss/dense_3_loss/Mean*
_output_shapes
: 

:training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/modFloorMod:training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/add;training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/Size*
T0*)
_class
loc:@loss/dense_3_loss/Mean*
_output_shapes
: 
Ќ
>training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/Shape_1Const*
valueB *
dtype0*)
_class
loc:@loss/dense_3_loss/Mean*
_output_shapes
: 
Џ
Btraining/RMSprop/gradients/loss/dense_3_loss/Mean_grad/range/startConst*
value	B : *
dtype0*)
_class
loc:@loss/dense_3_loss/Mean*
_output_shapes
: 
Џ
Btraining/RMSprop/gradients/loss/dense_3_loss/Mean_grad/range/deltaConst*
value	B :*
dtype0*)
_class
loc:@loss/dense_3_loss/Mean*
_output_shapes
: 
н
<training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/rangeRangeBtraining/RMSprop/gradients/loss/dense_3_loss/Mean_grad/range/start;training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/SizeBtraining/RMSprop/gradients/loss/dense_3_loss/Mean_grad/range/delta*

Tidx0*)
_class
loc:@loss/dense_3_loss/Mean*
_output_shapes
:
Ў
Atraining/RMSprop/gradients/loss/dense_3_loss/Mean_grad/Fill/valueConst*
value	B :*
dtype0*)
_class
loc:@loss/dense_3_loss/Mean*
_output_shapes
: 

;training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/FillFill>training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/Shape_1Atraining/RMSprop/gradients/loss/dense_3_loss/Mean_grad/Fill/value*
T0*)
_class
loc:@loss/dense_3_loss/Mean*
_output_shapes
: 
Ќ
Dtraining/RMSprop/gradients/loss/dense_3_loss/Mean_grad/DynamicStitchDynamicStitch<training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/range:training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/mod<training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/Shape;training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/Fill*
N*
T0*)
_class
loc:@loss/dense_3_loss/Mean*#
_output_shapes
:џџџџџџџџџ
­
@training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/Maximum/yConst*
value	B :*
dtype0*)
_class
loc:@loss/dense_3_loss/Mean*
_output_shapes
: 
Њ
>training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/MaximumMaximumDtraining/RMSprop/gradients/loss/dense_3_loss/Mean_grad/DynamicStitch@training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/Maximum/y*
T0*)
_class
loc:@loss/dense_3_loss/Mean*#
_output_shapes
:џџџџџџџџџ

?training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/floordivFloorDiv<training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/Shape>training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/Maximum*
T0*)
_class
loc:@loss/dense_3_loss/Mean*
_output_shapes
:
­
>training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/ReshapeReshape@training/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/truedivDtraining/RMSprop/gradients/loss/dense_3_loss/Mean_grad/DynamicStitch*
T0*
Tshape0*)
_class
loc:@loss/dense_3_loss/Mean*
_output_shapes
:
М
;training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/TileTile>training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/Reshape?training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/floordiv*
T0*

Tmultiples0*)
_class
loc:@loss/dense_3_loss/Mean*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
С
>training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/Shape_2Shapeloss/dense_3_loss/Square*
T0*
out_type0*)
_class
loc:@loss/dense_3_loss/Mean*
_output_shapes
:
П
>training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/Shape_3Shapeloss/dense_3_loss/Mean*
T0*
out_type0*)
_class
loc:@loss/dense_3_loss/Mean*
_output_shapes
:
Б
<training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/ConstConst*
valueB: *
dtype0*)
_class
loc:@loss/dense_3_loss/Mean*
_output_shapes
:
Њ
;training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/ProdProd>training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/Shape_2<training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/Const*
	keep_dims( *
T0*

Tidx0*)
_class
loc:@loss/dense_3_loss/Mean*
_output_shapes
: 
Г
>training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/Const_1Const*
valueB: *
dtype0*)
_class
loc:@loss/dense_3_loss/Mean*
_output_shapes
:
Ў
=training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/Prod_1Prod>training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/Shape_3>training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/Const_1*
	keep_dims( *
T0*

Tidx0*)
_class
loc:@loss/dense_3_loss/Mean*
_output_shapes
: 
Џ
Btraining/RMSprop/gradients/loss/dense_3_loss/Mean_grad/Maximum_1/yConst*
value	B :*
dtype0*)
_class
loc:@loss/dense_3_loss/Mean*
_output_shapes
: 

@training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/Maximum_1Maximum=training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/Prod_1Btraining/RMSprop/gradients/loss/dense_3_loss/Mean_grad/Maximum_1/y*
T0*)
_class
loc:@loss/dense_3_loss/Mean*
_output_shapes
: 

Atraining/RMSprop/gradients/loss/dense_3_loss/Mean_grad/floordiv_1FloorDiv;training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/Prod@training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/Maximum_1*
T0*)
_class
loc:@loss/dense_3_loss/Mean*
_output_shapes
: 
с
;training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/CastCastAtraining/RMSprop/gradients/loss/dense_3_loss/Mean_grad/floordiv_1*

SrcT0*

DstT0*)
_class
loc:@loss/dense_3_loss/Mean*
_output_shapes
: 
Љ
>training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/truedivRealDiv;training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/Tile;training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/Cast*
T0*)
_class
loc:@loss/dense_3_loss/Mean*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
ё
>training/RMSprop/gradients/loss/dense_3_loss/Square_grad/mul/xConst?^training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/truediv*
valueB
 *   @*
dtype0*+
_class!
loc:@loss/dense_3_loss/Square*
_output_shapes
: 

<training/RMSprop/gradients/loss/dense_3_loss/Square_grad/mulMul>training/RMSprop/gradients/loss/dense_3_loss/Square_grad/mul/xloss/dense_3_loss/sub*
T0*+
_class!
loc:@loss/dense_3_loss/Square*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
Ћ
>training/RMSprop/gradients/loss/dense_3_loss/Square_grad/mul_1Mul>training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/truediv<training/RMSprop/gradients/loss/dense_3_loss/Square_grad/mul*
T0*+
_class!
loc:@loss/dense_3_loss/Square*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
Д
;training/RMSprop/gradients/loss/dense_3_loss/sub_grad/ShapeShapedense_3/BiasAdd*
T0*
out_type0*(
_class
loc:@loss/dense_3_loss/sub*
_output_shapes
:
Е
=training/RMSprop/gradients/loss/dense_3_loss/sub_grad/Shape_1Shapedense_3_target*
T0*
out_type0*(
_class
loc:@loss/dense_3_loss/sub*
_output_shapes
:
Ч
Ktraining/RMSprop/gradients/loss/dense_3_loss/sub_grad/BroadcastGradientArgsBroadcastGradientArgs;training/RMSprop/gradients/loss/dense_3_loss/sub_grad/Shape=training/RMSprop/gradients/loss/dense_3_loss/sub_grad/Shape_1*
T0*(
_class
loc:@loss/dense_3_loss/sub*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
З
9training/RMSprop/gradients/loss/dense_3_loss/sub_grad/SumSum>training/RMSprop/gradients/loss/dense_3_loss/Square_grad/mul_1Ktraining/RMSprop/gradients/loss/dense_3_loss/sub_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*(
_class
loc:@loss/dense_3_loss/sub*
_output_shapes
:
Њ
=training/RMSprop/gradients/loss/dense_3_loss/sub_grad/ReshapeReshape9training/RMSprop/gradients/loss/dense_3_loss/sub_grad/Sum;training/RMSprop/gradients/loss/dense_3_loss/sub_grad/Shape*
T0*
Tshape0*(
_class
loc:@loss/dense_3_loss/sub*'
_output_shapes
:џџџџџџџџџ
Л
;training/RMSprop/gradients/loss/dense_3_loss/sub_grad/Sum_1Sum>training/RMSprop/gradients/loss/dense_3_loss/Square_grad/mul_1Mtraining/RMSprop/gradients/loss/dense_3_loss/sub_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*(
_class
loc:@loss/dense_3_loss/sub*
_output_shapes
:
Ъ
9training/RMSprop/gradients/loss/dense_3_loss/sub_grad/NegNeg;training/RMSprop/gradients/loss/dense_3_loss/sub_grad/Sum_1*
T0*(
_class
loc:@loss/dense_3_loss/sub*
_output_shapes
:
З
?training/RMSprop/gradients/loss/dense_3_loss/sub_grad/Reshape_1Reshape9training/RMSprop/gradients/loss/dense_3_loss/sub_grad/Neg=training/RMSprop/gradients/loss/dense_3_loss/sub_grad/Shape_1*
T0*
Tshape0*(
_class
loc:@loss/dense_3_loss/sub*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
щ
;training/RMSprop/gradients/dense_3/BiasAdd_grad/BiasAddGradBiasAddGrad=training/RMSprop/gradients/loss/dense_3_loss/sub_grad/Reshape*
T0*
data_formatNHWC*"
_class
loc:@dense_3/BiasAdd*
_output_shapes
:

5training/RMSprop/gradients/dense_3/MatMul_grad/MatMulMatMul=training/RMSprop/gradients/loss/dense_3_loss/sub_grad/Reshapedense_3/kernel/read*
transpose_a( *
transpose_b(*
T0*!
_class
loc:@dense_3/MatMul*'
_output_shapes
:џџџџџџџџџ@

7training/RMSprop/gradients/dense_3/MatMul_grad/MatMul_1MatMuldense_2/Relu=training/RMSprop/gradients/loss/dense_3_loss/sub_grad/Reshape*
transpose_a(*
transpose_b( *
T0*!
_class
loc:@dense_3/MatMul*
_output_shapes

:@
й
5training/RMSprop/gradients/dense_2/Relu_grad/ReluGradReluGrad5training/RMSprop/gradients/dense_3/MatMul_grad/MatMuldense_2/Relu*
T0*
_class
loc:@dense_2/Relu*'
_output_shapes
:џџџџџџџџџ@
с
;training/RMSprop/gradients/dense_2/BiasAdd_grad/BiasAddGradBiasAddGrad5training/RMSprop/gradients/dense_2/Relu_grad/ReluGrad*
T0*
data_formatNHWC*"
_class
loc:@dense_2/BiasAdd*
_output_shapes
:@

5training/RMSprop/gradients/dense_2/MatMul_grad/MatMulMatMul5training/RMSprop/gradients/dense_2/Relu_grad/ReluGraddense_2/kernel/read*
transpose_a( *
transpose_b(*
T0*!
_class
loc:@dense_2/MatMul*'
_output_shapes
:џџџџџџџџџ@
ј
7training/RMSprop/gradients/dense_2/MatMul_grad/MatMul_1MatMuldense_1/Relu5training/RMSprop/gradients/dense_2/Relu_grad/ReluGrad*
transpose_a(*
transpose_b( *
T0*!
_class
loc:@dense_2/MatMul*
_output_shapes

:@@
й
5training/RMSprop/gradients/dense_1/Relu_grad/ReluGradReluGrad5training/RMSprop/gradients/dense_2/MatMul_grad/MatMuldense_1/Relu*
T0*
_class
loc:@dense_1/Relu*'
_output_shapes
:џџџџџџџџџ@
с
;training/RMSprop/gradients/dense_1/BiasAdd_grad/BiasAddGradBiasAddGrad5training/RMSprop/gradients/dense_1/Relu_grad/ReluGrad*
T0*
data_formatNHWC*"
_class
loc:@dense_1/BiasAdd*
_output_shapes
:@

5training/RMSprop/gradients/dense_1/MatMul_grad/MatMulMatMul5training/RMSprop/gradients/dense_1/Relu_grad/ReluGraddense_1/kernel/read*
transpose_a( *
transpose_b(*
T0*!
_class
loc:@dense_1/MatMul*'
_output_shapes
:џџџџџџџџџ
љ
7training/RMSprop/gradients/dense_1/MatMul_grad/MatMul_1MatMuldense_1_input5training/RMSprop/gradients/dense_1/Relu_grad/ReluGrad*
transpose_a(*
transpose_b( *
T0*!
_class
loc:@dense_1/MatMul*
_output_shapes

:@
k
training/RMSprop/ConstConst*
valueB@*    *
dtype0*
_output_shapes

:@

training/RMSprop/Variable
VariableV2*
shape
:@*
dtype0*
	container *
shared_name *
_output_shapes

:@
н
 training/RMSprop/Variable/AssignAssigntraining/RMSprop/Variabletraining/RMSprop/Const*
T0*
validate_shape(*
use_locking(*,
_class"
 loc:@training/RMSprop/Variable*
_output_shapes

:@

training/RMSprop/Variable/readIdentitytraining/RMSprop/Variable*
T0*,
_class"
 loc:@training/RMSprop/Variable*
_output_shapes

:@
e
training/RMSprop/Const_1Const*
valueB@*    *
dtype0*
_output_shapes
:@

training/RMSprop/Variable_1
VariableV2*
shape:@*
dtype0*
	container *
shared_name *
_output_shapes
:@
с
"training/RMSprop/Variable_1/AssignAssigntraining/RMSprop/Variable_1training/RMSprop/Const_1*
T0*
validate_shape(*
use_locking(*.
_class$
" loc:@training/RMSprop/Variable_1*
_output_shapes
:@

 training/RMSprop/Variable_1/readIdentitytraining/RMSprop/Variable_1*
T0*.
_class$
" loc:@training/RMSprop/Variable_1*
_output_shapes
:@
m
training/RMSprop/Const_2Const*
valueB@@*    *
dtype0*
_output_shapes

:@@

training/RMSprop/Variable_2
VariableV2*
shape
:@@*
dtype0*
	container *
shared_name *
_output_shapes

:@@
х
"training/RMSprop/Variable_2/AssignAssigntraining/RMSprop/Variable_2training/RMSprop/Const_2*
T0*
validate_shape(*
use_locking(*.
_class$
" loc:@training/RMSprop/Variable_2*
_output_shapes

:@@
Ђ
 training/RMSprop/Variable_2/readIdentitytraining/RMSprop/Variable_2*
T0*.
_class$
" loc:@training/RMSprop/Variable_2*
_output_shapes

:@@
e
training/RMSprop/Const_3Const*
valueB@*    *
dtype0*
_output_shapes
:@

training/RMSprop/Variable_3
VariableV2*
shape:@*
dtype0*
	container *
shared_name *
_output_shapes
:@
с
"training/RMSprop/Variable_3/AssignAssigntraining/RMSprop/Variable_3training/RMSprop/Const_3*
T0*
validate_shape(*
use_locking(*.
_class$
" loc:@training/RMSprop/Variable_3*
_output_shapes
:@

 training/RMSprop/Variable_3/readIdentitytraining/RMSprop/Variable_3*
T0*.
_class$
" loc:@training/RMSprop/Variable_3*
_output_shapes
:@
m
training/RMSprop/Const_4Const*
valueB@*    *
dtype0*
_output_shapes

:@

training/RMSprop/Variable_4
VariableV2*
shape
:@*
dtype0*
	container *
shared_name *
_output_shapes

:@
х
"training/RMSprop/Variable_4/AssignAssigntraining/RMSprop/Variable_4training/RMSprop/Const_4*
T0*
validate_shape(*
use_locking(*.
_class$
" loc:@training/RMSprop/Variable_4*
_output_shapes

:@
Ђ
 training/RMSprop/Variable_4/readIdentitytraining/RMSprop/Variable_4*
T0*.
_class$
" loc:@training/RMSprop/Variable_4*
_output_shapes

:@
e
training/RMSprop/Const_5Const*
valueB*    *
dtype0*
_output_shapes
:

training/RMSprop/Variable_5
VariableV2*
shape:*
dtype0*
	container *
shared_name *
_output_shapes
:
с
"training/RMSprop/Variable_5/AssignAssigntraining/RMSprop/Variable_5training/RMSprop/Const_5*
T0*
validate_shape(*
use_locking(*.
_class$
" loc:@training/RMSprop/Variable_5*
_output_shapes
:

 training/RMSprop/Variable_5/readIdentitytraining/RMSprop/Variable_5*
T0*.
_class$
" loc:@training/RMSprop/Variable_5*
_output_shapes
:
b
 training/RMSprop/AssignAdd/valueConst*
value	B	 R*
dtype0	*
_output_shapes
: 
И
training/RMSprop/AssignAdd	AssignAddRMSprop/iterations training/RMSprop/AssignAdd/value*
T0	*
use_locking( *%
_class
loc:@RMSprop/iterations*
_output_shapes
: 
v
training/RMSprop/mulMulRMSprop/rho/readtraining/RMSprop/Variable/read*
T0*
_output_shapes

:@
[
training/RMSprop/sub/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/RMSprop/subSubtraining/RMSprop/sub/xRMSprop/rho/read*
T0*
_output_shapes
: 

training/RMSprop/SquareSquare7training/RMSprop/gradients/dense_1/MatMul_grad/MatMul_1*
T0*
_output_shapes

:@
u
training/RMSprop/mul_1Multraining/RMSprop/subtraining/RMSprop/Square*
T0*
_output_shapes

:@
r
training/RMSprop/addAddtraining/RMSprop/multraining/RMSprop/mul_1*
T0*
_output_shapes

:@
в
training/RMSprop/AssignAssigntraining/RMSprop/Variabletraining/RMSprop/add*
T0*
validate_shape(*
use_locking(*,
_class"
 loc:@training/RMSprop/Variable*
_output_shapes

:@

training/RMSprop/mul_2MulRMSprop/lr/read7training/RMSprop/gradients/dense_1/MatMul_grad/MatMul_1*
T0*
_output_shapes

:@
]
training/RMSprop/Const_6Const*
valueB
 *    *
dtype0*
_output_shapes
: 
]
training/RMSprop/Const_7Const*
valueB
 *  *
dtype0*
_output_shapes
: 

&training/RMSprop/clip_by_value/MinimumMinimumtraining/RMSprop/addtraining/RMSprop/Const_7*
T0*
_output_shapes

:@

training/RMSprop/clip_by_valueMaximum&training/RMSprop/clip_by_value/Minimumtraining/RMSprop/Const_6*
T0*
_output_shapes

:@
f
training/RMSprop/SqrtSqrttraining/RMSprop/clip_by_value*
T0*
_output_shapes

:@
]
training/RMSprop/add_1/yConst*
valueB
 *wЬ+2*
dtype0*
_output_shapes
: 
w
training/RMSprop/add_1Addtraining/RMSprop/Sqrttraining/RMSprop/add_1/y*
T0*
_output_shapes

:@
|
training/RMSprop/truedivRealDivtraining/RMSprop/mul_2training/RMSprop/add_1*
T0*
_output_shapes

:@
u
training/RMSprop/sub_1Subdense_1/kernel/readtraining/RMSprop/truediv*
T0*
_output_shapes

:@
Р
training/RMSprop/Assign_1Assigndense_1/kerneltraining/RMSprop/sub_1*
T0*
validate_shape(*
use_locking(*!
_class
loc:@dense_1/kernel*
_output_shapes

:@
v
training/RMSprop/mul_3MulRMSprop/rho/read training/RMSprop/Variable_1/read*
T0*
_output_shapes
:@
]
training/RMSprop/sub_2/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
j
training/RMSprop/sub_2Subtraining/RMSprop/sub_2/xRMSprop/rho/read*
T0*
_output_shapes
: 

training/RMSprop/Square_1Square;training/RMSprop/gradients/dense_1/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:@
u
training/RMSprop/mul_4Multraining/RMSprop/sub_2training/RMSprop/Square_1*
T0*
_output_shapes
:@
r
training/RMSprop/add_2Addtraining/RMSprop/mul_3training/RMSprop/mul_4*
T0*
_output_shapes
:@
ж
training/RMSprop/Assign_2Assigntraining/RMSprop/Variable_1training/RMSprop/add_2*
T0*
validate_shape(*
use_locking(*.
_class$
" loc:@training/RMSprop/Variable_1*
_output_shapes
:@

training/RMSprop/mul_5MulRMSprop/lr/read;training/RMSprop/gradients/dense_1/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:@
]
training/RMSprop/Const_8Const*
valueB
 *    *
dtype0*
_output_shapes
: 
]
training/RMSprop/Const_9Const*
valueB
 *  *
dtype0*
_output_shapes
: 

(training/RMSprop/clip_by_value_1/MinimumMinimumtraining/RMSprop/add_2training/RMSprop/Const_9*
T0*
_output_shapes
:@

 training/RMSprop/clip_by_value_1Maximum(training/RMSprop/clip_by_value_1/Minimumtraining/RMSprop/Const_8*
T0*
_output_shapes
:@
f
training/RMSprop/Sqrt_1Sqrt training/RMSprop/clip_by_value_1*
T0*
_output_shapes
:@
]
training/RMSprop/add_3/yConst*
valueB
 *wЬ+2*
dtype0*
_output_shapes
: 
u
training/RMSprop/add_3Addtraining/RMSprop/Sqrt_1training/RMSprop/add_3/y*
T0*
_output_shapes
:@
z
training/RMSprop/truediv_1RealDivtraining/RMSprop/mul_5training/RMSprop/add_3*
T0*
_output_shapes
:@
q
training/RMSprop/sub_3Subdense_1/bias/readtraining/RMSprop/truediv_1*
T0*
_output_shapes
:@
И
training/RMSprop/Assign_3Assigndense_1/biastraining/RMSprop/sub_3*
T0*
validate_shape(*
use_locking(*
_class
loc:@dense_1/bias*
_output_shapes
:@
z
training/RMSprop/mul_6MulRMSprop/rho/read training/RMSprop/Variable_2/read*
T0*
_output_shapes

:@@
]
training/RMSprop/sub_4/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
j
training/RMSprop/sub_4Subtraining/RMSprop/sub_4/xRMSprop/rho/read*
T0*
_output_shapes
: 

training/RMSprop/Square_2Square7training/RMSprop/gradients/dense_2/MatMul_grad/MatMul_1*
T0*
_output_shapes

:@@
y
training/RMSprop/mul_7Multraining/RMSprop/sub_4training/RMSprop/Square_2*
T0*
_output_shapes

:@@
v
training/RMSprop/add_4Addtraining/RMSprop/mul_6training/RMSprop/mul_7*
T0*
_output_shapes

:@@
к
training/RMSprop/Assign_4Assigntraining/RMSprop/Variable_2training/RMSprop/add_4*
T0*
validate_shape(*
use_locking(*.
_class$
" loc:@training/RMSprop/Variable_2*
_output_shapes

:@@

training/RMSprop/mul_8MulRMSprop/lr/read7training/RMSprop/gradients/dense_2/MatMul_grad/MatMul_1*
T0*
_output_shapes

:@@
^
training/RMSprop/Const_10Const*
valueB
 *    *
dtype0*
_output_shapes
: 
^
training/RMSprop/Const_11Const*
valueB
 *  *
dtype0*
_output_shapes
: 

(training/RMSprop/clip_by_value_2/MinimumMinimumtraining/RMSprop/add_4training/RMSprop/Const_11*
T0*
_output_shapes

:@@

 training/RMSprop/clip_by_value_2Maximum(training/RMSprop/clip_by_value_2/Minimumtraining/RMSprop/Const_10*
T0*
_output_shapes

:@@
j
training/RMSprop/Sqrt_2Sqrt training/RMSprop/clip_by_value_2*
T0*
_output_shapes

:@@
]
training/RMSprop/add_5/yConst*
valueB
 *wЬ+2*
dtype0*
_output_shapes
: 
y
training/RMSprop/add_5Addtraining/RMSprop/Sqrt_2training/RMSprop/add_5/y*
T0*
_output_shapes

:@@
~
training/RMSprop/truediv_2RealDivtraining/RMSprop/mul_8training/RMSprop/add_5*
T0*
_output_shapes

:@@
w
training/RMSprop/sub_5Subdense_2/kernel/readtraining/RMSprop/truediv_2*
T0*
_output_shapes

:@@
Р
training/RMSprop/Assign_5Assigndense_2/kerneltraining/RMSprop/sub_5*
T0*
validate_shape(*
use_locking(*!
_class
loc:@dense_2/kernel*
_output_shapes

:@@
v
training/RMSprop/mul_9MulRMSprop/rho/read training/RMSprop/Variable_3/read*
T0*
_output_shapes
:@
]
training/RMSprop/sub_6/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
j
training/RMSprop/sub_6Subtraining/RMSprop/sub_6/xRMSprop/rho/read*
T0*
_output_shapes
: 

training/RMSprop/Square_3Square;training/RMSprop/gradients/dense_2/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:@
v
training/RMSprop/mul_10Multraining/RMSprop/sub_6training/RMSprop/Square_3*
T0*
_output_shapes
:@
s
training/RMSprop/add_6Addtraining/RMSprop/mul_9training/RMSprop/mul_10*
T0*
_output_shapes
:@
ж
training/RMSprop/Assign_6Assigntraining/RMSprop/Variable_3training/RMSprop/add_6*
T0*
validate_shape(*
use_locking(*.
_class$
" loc:@training/RMSprop/Variable_3*
_output_shapes
:@

training/RMSprop/mul_11MulRMSprop/lr/read;training/RMSprop/gradients/dense_2/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:@
^
training/RMSprop/Const_12Const*
valueB
 *    *
dtype0*
_output_shapes
: 
^
training/RMSprop/Const_13Const*
valueB
 *  *
dtype0*
_output_shapes
: 

(training/RMSprop/clip_by_value_3/MinimumMinimumtraining/RMSprop/add_6training/RMSprop/Const_13*
T0*
_output_shapes
:@

 training/RMSprop/clip_by_value_3Maximum(training/RMSprop/clip_by_value_3/Minimumtraining/RMSprop/Const_12*
T0*
_output_shapes
:@
f
training/RMSprop/Sqrt_3Sqrt training/RMSprop/clip_by_value_3*
T0*
_output_shapes
:@
]
training/RMSprop/add_7/yConst*
valueB
 *wЬ+2*
dtype0*
_output_shapes
: 
u
training/RMSprop/add_7Addtraining/RMSprop/Sqrt_3training/RMSprop/add_7/y*
T0*
_output_shapes
:@
{
training/RMSprop/truediv_3RealDivtraining/RMSprop/mul_11training/RMSprop/add_7*
T0*
_output_shapes
:@
q
training/RMSprop/sub_7Subdense_2/bias/readtraining/RMSprop/truediv_3*
T0*
_output_shapes
:@
И
training/RMSprop/Assign_7Assigndense_2/biastraining/RMSprop/sub_7*
T0*
validate_shape(*
use_locking(*
_class
loc:@dense_2/bias*
_output_shapes
:@
{
training/RMSprop/mul_12MulRMSprop/rho/read training/RMSprop/Variable_4/read*
T0*
_output_shapes

:@
]
training/RMSprop/sub_8/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
j
training/RMSprop/sub_8Subtraining/RMSprop/sub_8/xRMSprop/rho/read*
T0*
_output_shapes
: 

training/RMSprop/Square_4Square7training/RMSprop/gradients/dense_3/MatMul_grad/MatMul_1*
T0*
_output_shapes

:@
z
training/RMSprop/mul_13Multraining/RMSprop/sub_8training/RMSprop/Square_4*
T0*
_output_shapes

:@
x
training/RMSprop/add_8Addtraining/RMSprop/mul_12training/RMSprop/mul_13*
T0*
_output_shapes

:@
к
training/RMSprop/Assign_8Assigntraining/RMSprop/Variable_4training/RMSprop/add_8*
T0*
validate_shape(*
use_locking(*.
_class$
" loc:@training/RMSprop/Variable_4*
_output_shapes

:@

training/RMSprop/mul_14MulRMSprop/lr/read7training/RMSprop/gradients/dense_3/MatMul_grad/MatMul_1*
T0*
_output_shapes

:@
^
training/RMSprop/Const_14Const*
valueB
 *    *
dtype0*
_output_shapes
: 
^
training/RMSprop/Const_15Const*
valueB
 *  *
dtype0*
_output_shapes
: 

(training/RMSprop/clip_by_value_4/MinimumMinimumtraining/RMSprop/add_8training/RMSprop/Const_15*
T0*
_output_shapes

:@

 training/RMSprop/clip_by_value_4Maximum(training/RMSprop/clip_by_value_4/Minimumtraining/RMSprop/Const_14*
T0*
_output_shapes

:@
j
training/RMSprop/Sqrt_4Sqrt training/RMSprop/clip_by_value_4*
T0*
_output_shapes

:@
]
training/RMSprop/add_9/yConst*
valueB
 *wЬ+2*
dtype0*
_output_shapes
: 
y
training/RMSprop/add_9Addtraining/RMSprop/Sqrt_4training/RMSprop/add_9/y*
T0*
_output_shapes

:@

training/RMSprop/truediv_4RealDivtraining/RMSprop/mul_14training/RMSprop/add_9*
T0*
_output_shapes

:@
w
training/RMSprop/sub_9Subdense_3/kernel/readtraining/RMSprop/truediv_4*
T0*
_output_shapes

:@
Р
training/RMSprop/Assign_9Assigndense_3/kerneltraining/RMSprop/sub_9*
T0*
validate_shape(*
use_locking(*!
_class
loc:@dense_3/kernel*
_output_shapes

:@
w
training/RMSprop/mul_15MulRMSprop/rho/read training/RMSprop/Variable_5/read*
T0*
_output_shapes
:
^
training/RMSprop/sub_10/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
l
training/RMSprop/sub_10Subtraining/RMSprop/sub_10/xRMSprop/rho/read*
T0*
_output_shapes
: 

training/RMSprop/Square_5Square;training/RMSprop/gradients/dense_3/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:
w
training/RMSprop/mul_16Multraining/RMSprop/sub_10training/RMSprop/Square_5*
T0*
_output_shapes
:
u
training/RMSprop/add_10Addtraining/RMSprop/mul_15training/RMSprop/mul_16*
T0*
_output_shapes
:
и
training/RMSprop/Assign_10Assigntraining/RMSprop/Variable_5training/RMSprop/add_10*
T0*
validate_shape(*
use_locking(*.
_class$
" loc:@training/RMSprop/Variable_5*
_output_shapes
:

training/RMSprop/mul_17MulRMSprop/lr/read;training/RMSprop/gradients/dense_3/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:
^
training/RMSprop/Const_16Const*
valueB
 *    *
dtype0*
_output_shapes
: 
^
training/RMSprop/Const_17Const*
valueB
 *  *
dtype0*
_output_shapes
: 

(training/RMSprop/clip_by_value_5/MinimumMinimumtraining/RMSprop/add_10training/RMSprop/Const_17*
T0*
_output_shapes
:

 training/RMSprop/clip_by_value_5Maximum(training/RMSprop/clip_by_value_5/Minimumtraining/RMSprop/Const_16*
T0*
_output_shapes
:
f
training/RMSprop/Sqrt_5Sqrt training/RMSprop/clip_by_value_5*
T0*
_output_shapes
:
^
training/RMSprop/add_11/yConst*
valueB
 *wЬ+2*
dtype0*
_output_shapes
: 
w
training/RMSprop/add_11Addtraining/RMSprop/Sqrt_5training/RMSprop/add_11/y*
T0*
_output_shapes
:
|
training/RMSprop/truediv_5RealDivtraining/RMSprop/mul_17training/RMSprop/add_11*
T0*
_output_shapes
:
r
training/RMSprop/sub_11Subdense_3/bias/readtraining/RMSprop/truediv_5*
T0*
_output_shapes
:
К
training/RMSprop/Assign_11Assigndense_3/biastraining/RMSprop/sub_11*
T0*
validate_shape(*
use_locking(*
_class
loc:@dense_3/bias*
_output_shapes
:
И
training/group_depsNoOp	^loss/mul#^metrics/mean_absolute_error/Mean_1^training/RMSprop/AssignAdd^training/RMSprop/Assign^training/RMSprop/Assign_1^training/RMSprop/Assign_2^training/RMSprop/Assign_3^training/RMSprop/Assign_4^training/RMSprop/Assign_5^training/RMSprop/Assign_6^training/RMSprop/Assign_7^training/RMSprop/Assign_8^training/RMSprop/Assign_9^training/RMSprop/Assign_10^training/RMSprop/Assign_11
B

group_depsNoOp	^loss/mul#^metrics/mean_absolute_error/Mean_1

IsVariableInitializedIsVariableInitializeddense_1/kernel*
dtype0*!
_class
loc:@dense_1/kernel*
_output_shapes
: 

IsVariableInitialized_1IsVariableInitializeddense_1/bias*
dtype0*
_class
loc:@dense_1/bias*
_output_shapes
: 

IsVariableInitialized_2IsVariableInitializeddense_2/kernel*
dtype0*!
_class
loc:@dense_2/kernel*
_output_shapes
: 

IsVariableInitialized_3IsVariableInitializeddense_2/bias*
dtype0*
_class
loc:@dense_2/bias*
_output_shapes
: 

IsVariableInitialized_4IsVariableInitializeddense_3/kernel*
dtype0*!
_class
loc:@dense_3/kernel*
_output_shapes
: 

IsVariableInitialized_5IsVariableInitializeddense_3/bias*
dtype0*
_class
loc:@dense_3/bias*
_output_shapes
: 

IsVariableInitialized_6IsVariableInitialized
RMSprop/lr*
dtype0*
_class
loc:@RMSprop/lr*
_output_shapes
: 

IsVariableInitialized_7IsVariableInitializedRMSprop/rho*
dtype0*
_class
loc:@RMSprop/rho*
_output_shapes
: 

IsVariableInitialized_8IsVariableInitializedRMSprop/decay*
dtype0* 
_class
loc:@RMSprop/decay*
_output_shapes
: 

IsVariableInitialized_9IsVariableInitializedRMSprop/iterations*
dtype0	*%
_class
loc:@RMSprop/iterations*
_output_shapes
: 

IsVariableInitialized_10IsVariableInitializedtraining/RMSprop/Variable*
dtype0*,
_class"
 loc:@training/RMSprop/Variable*
_output_shapes
: 
Ѓ
IsVariableInitialized_11IsVariableInitializedtraining/RMSprop/Variable_1*
dtype0*.
_class$
" loc:@training/RMSprop/Variable_1*
_output_shapes
: 
Ѓ
IsVariableInitialized_12IsVariableInitializedtraining/RMSprop/Variable_2*
dtype0*.
_class$
" loc:@training/RMSprop/Variable_2*
_output_shapes
: 
Ѓ
IsVariableInitialized_13IsVariableInitializedtraining/RMSprop/Variable_3*
dtype0*.
_class$
" loc:@training/RMSprop/Variable_3*
_output_shapes
: 
Ѓ
IsVariableInitialized_14IsVariableInitializedtraining/RMSprop/Variable_4*
dtype0*.
_class$
" loc:@training/RMSprop/Variable_4*
_output_shapes
: 
Ѓ
IsVariableInitialized_15IsVariableInitializedtraining/RMSprop/Variable_5*
dtype0*.
_class$
" loc:@training/RMSprop/Variable_5*
_output_shapes
: 
Ю
initNoOp^dense_1/kernel/Assign^dense_1/bias/Assign^dense_2/kernel/Assign^dense_2/bias/Assign^dense_3/kernel/Assign^dense_3/bias/Assign^RMSprop/lr/Assign^RMSprop/rho/Assign^RMSprop/decay/Assign^RMSprop/iterations/Assign!^training/RMSprop/Variable/Assign#^training/RMSprop/Variable_1/Assign#^training/RMSprop/Variable_2/Assign#^training/RMSprop/Variable_3/Assign#^training/RMSprop/Variable_4/Assign#^training/RMSprop/Variable_5/Assign
e
dense_1/kernel_0/tagConst*!
valueB Bdense_1/kernel_0*
dtype0*
_output_shapes
: 
p
dense_1/kernel_0HistogramSummarydense_1/kernel_0/tagdense_1/kernel/read*
T0*
_output_shapes
: 
a
dense_1/bias_0/tagConst*
valueB Bdense_1/bias_0*
dtype0*
_output_shapes
: 
j
dense_1/bias_0HistogramSummarydense_1/bias_0/tagdense_1/bias/read*
T0*
_output_shapes
: 
[
dense_1_out/tagConst*
valueB Bdense_1_out*
dtype0*
_output_shapes
: 
_
dense_1_outHistogramSummarydense_1_out/tagdense_1/Relu*
T0*
_output_shapes
: 
e
dense_2/kernel_0/tagConst*!
valueB Bdense_2/kernel_0*
dtype0*
_output_shapes
: 
p
dense_2/kernel_0HistogramSummarydense_2/kernel_0/tagdense_2/kernel/read*
T0*
_output_shapes
: 
a
dense_2/bias_0/tagConst*
valueB Bdense_2/bias_0*
dtype0*
_output_shapes
: 
j
dense_2/bias_0HistogramSummarydense_2/bias_0/tagdense_2/bias/read*
T0*
_output_shapes
: 
[
dense_2_out/tagConst*
valueB Bdense_2_out*
dtype0*
_output_shapes
: 
_
dense_2_outHistogramSummarydense_2_out/tagdense_2/Relu*
T0*
_output_shapes
: 
e
dense_3/kernel_0/tagConst*!
valueB Bdense_3/kernel_0*
dtype0*
_output_shapes
: 
p
dense_3/kernel_0HistogramSummarydense_3/kernel_0/tagdense_3/kernel/read*
T0*
_output_shapes
: 
a
dense_3/bias_0/tagConst*
valueB Bdense_3/bias_0*
dtype0*
_output_shapes
: 
j
dense_3/bias_0HistogramSummarydense_3/bias_0/tagdense_3/bias/read*
T0*
_output_shapes
: 
[
dense_3_out/tagConst*
valueB Bdense_3_out*
dtype0*
_output_shapes
: 
b
dense_3_outHistogramSummarydense_3_out/tagdense_3/BiasAdd*
T0*
_output_shapes
: 
а
Merge/MergeSummaryMergeSummarydense_1/kernel_0dense_1/bias_0dense_1_outdense_2/kernel_0dense_2/bias_0dense_2_outdense_3/kernel_0dense_3/bias_0dense_3_out*
N	*
_output_shapes
: 
p
dense_4_inputPlaceholder*
dtype0*
shape:џџџџџџџџџ*'
_output_shapes
:џџџџџџџџџ
m
dense_4/random_uniform/shapeConst*
valueB"   @   *
dtype0*
_output_shapes
:
_
dense_4/random_uniform/minConst*
valueB
 *!ьО*
dtype0*
_output_shapes
: 
_
dense_4/random_uniform/maxConst*
valueB
 *!ь>*
dtype0*
_output_shapes
: 
Ї
$dense_4/random_uniform/RandomUniformRandomUniformdense_4/random_uniform/shape*
seedБџх)*
seed2ЈшD*
dtype0*
T0*
_output_shapes

:@
z
dense_4/random_uniform/subSubdense_4/random_uniform/maxdense_4/random_uniform/min*
T0*
_output_shapes
: 

dense_4/random_uniform/mulMul$dense_4/random_uniform/RandomUniformdense_4/random_uniform/sub*
T0*
_output_shapes

:@
~
dense_4/random_uniformAdddense_4/random_uniform/muldense_4/random_uniform/min*
T0*
_output_shapes

:@

dense_4/kernel
VariableV2*
shape
:@*
dtype0*
	container *
shared_name *
_output_shapes

:@
М
dense_4/kernel/AssignAssigndense_4/kerneldense_4/random_uniform*
T0*
validate_shape(*
use_locking(*!
_class
loc:@dense_4/kernel*
_output_shapes

:@
{
dense_4/kernel/readIdentitydense_4/kernel*
T0*!
_class
loc:@dense_4/kernel*
_output_shapes

:@
Z
dense_4/ConstConst*
valueB@*    *
dtype0*
_output_shapes
:@
x
dense_4/bias
VariableV2*
shape:@*
dtype0*
	container *
shared_name *
_output_shapes
:@
Љ
dense_4/bias/AssignAssigndense_4/biasdense_4/Const*
T0*
validate_shape(*
use_locking(*
_class
loc:@dense_4/bias*
_output_shapes
:@
q
dense_4/bias/readIdentitydense_4/bias*
T0*
_class
loc:@dense_4/bias*
_output_shapes
:@

dense_4/MatMulMatMuldense_4_inputdense_4/kernel/read*
transpose_a( *
transpose_b( *
T0*'
_output_shapes
:џџџџџџџџџ@

dense_4/BiasAddBiasAdddense_4/MatMuldense_4/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:џџџџџџџџџ@
W
dense_4/ReluReludense_4/BiasAdd*
T0*'
_output_shapes
:џџџџџџџџџ@
m
dense_5/random_uniform/shapeConst*
valueB"@   @   *
dtype0*
_output_shapes
:
_
dense_5/random_uniform/minConst*
valueB
 *зГ]О*
dtype0*
_output_shapes
: 
_
dense_5/random_uniform/maxConst*
valueB
 *зГ]>*
dtype0*
_output_shapes
: 
Ї
$dense_5/random_uniform/RandomUniformRandomUniformdense_5/random_uniform/shape*
seedБџх)*
seed2їн *
dtype0*
T0*
_output_shapes

:@@
z
dense_5/random_uniform/subSubdense_5/random_uniform/maxdense_5/random_uniform/min*
T0*
_output_shapes
: 

dense_5/random_uniform/mulMul$dense_5/random_uniform/RandomUniformdense_5/random_uniform/sub*
T0*
_output_shapes

:@@
~
dense_5/random_uniformAdddense_5/random_uniform/muldense_5/random_uniform/min*
T0*
_output_shapes

:@@

dense_5/kernel
VariableV2*
shape
:@@*
dtype0*
	container *
shared_name *
_output_shapes

:@@
М
dense_5/kernel/AssignAssigndense_5/kerneldense_5/random_uniform*
T0*
validate_shape(*
use_locking(*!
_class
loc:@dense_5/kernel*
_output_shapes

:@@
{
dense_5/kernel/readIdentitydense_5/kernel*
T0*!
_class
loc:@dense_5/kernel*
_output_shapes

:@@
Z
dense_5/ConstConst*
valueB@*    *
dtype0*
_output_shapes
:@
x
dense_5/bias
VariableV2*
shape:@*
dtype0*
	container *
shared_name *
_output_shapes
:@
Љ
dense_5/bias/AssignAssigndense_5/biasdense_5/Const*
T0*
validate_shape(*
use_locking(*
_class
loc:@dense_5/bias*
_output_shapes
:@
q
dense_5/bias/readIdentitydense_5/bias*
T0*
_class
loc:@dense_5/bias*
_output_shapes
:@

dense_5/MatMulMatMuldense_4/Reludense_5/kernel/read*
transpose_a( *
transpose_b( *
T0*'
_output_shapes
:џџџџџџџџџ@

dense_5/BiasAddBiasAdddense_5/MatMuldense_5/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:џџџџџџџџџ@
W
dense_5/ReluReludense_5/BiasAdd*
T0*'
_output_shapes
:џџџџџџџџџ@
m
dense_6/random_uniform/shapeConst*
valueB"@      *
dtype0*
_output_shapes
:
_
dense_6/random_uniform/minConst*
valueB
 *О*
dtype0*
_output_shapes
: 
_
dense_6/random_uniform/maxConst*
valueB
 *>*
dtype0*
_output_shapes
: 
Ј
$dense_6/random_uniform/RandomUniformRandomUniformdense_6/random_uniform/shape*
seedБџх)*
seed2мњ*
dtype0*
T0*
_output_shapes

:@
z
dense_6/random_uniform/subSubdense_6/random_uniform/maxdense_6/random_uniform/min*
T0*
_output_shapes
: 

dense_6/random_uniform/mulMul$dense_6/random_uniform/RandomUniformdense_6/random_uniform/sub*
T0*
_output_shapes

:@
~
dense_6/random_uniformAdddense_6/random_uniform/muldense_6/random_uniform/min*
T0*
_output_shapes

:@

dense_6/kernel
VariableV2*
shape
:@*
dtype0*
	container *
shared_name *
_output_shapes

:@
М
dense_6/kernel/AssignAssigndense_6/kerneldense_6/random_uniform*
T0*
validate_shape(*
use_locking(*!
_class
loc:@dense_6/kernel*
_output_shapes

:@
{
dense_6/kernel/readIdentitydense_6/kernel*
T0*!
_class
loc:@dense_6/kernel*
_output_shapes

:@
Z
dense_6/ConstConst*
valueB*    *
dtype0*
_output_shapes
:
x
dense_6/bias
VariableV2*
shape:*
dtype0*
	container *
shared_name *
_output_shapes
:
Љ
dense_6/bias/AssignAssigndense_6/biasdense_6/Const*
T0*
validate_shape(*
use_locking(*
_class
loc:@dense_6/bias*
_output_shapes
:
q
dense_6/bias/readIdentitydense_6/bias*
T0*
_class
loc:@dense_6/bias*
_output_shapes
:

dense_6/MatMulMatMuldense_5/Reludense_6/kernel/read*
transpose_a( *
transpose_b( *
T0*'
_output_shapes
:џџџџџџџџџ

dense_6/BiasAddBiasAdddense_6/MatMuldense_6/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:џџџџџџџџџ
_
RMSprop_1/lr/initial_valueConst*
valueB
 *o:*
dtype0*
_output_shapes
: 
p
RMSprop_1/lr
VariableV2*
shape: *
dtype0*
	container *
shared_name *
_output_shapes
: 
В
RMSprop_1/lr/AssignAssignRMSprop_1/lrRMSprop_1/lr/initial_value*
T0*
validate_shape(*
use_locking(*
_class
loc:@RMSprop_1/lr*
_output_shapes
: 
m
RMSprop_1/lr/readIdentityRMSprop_1/lr*
T0*
_class
loc:@RMSprop_1/lr*
_output_shapes
: 
`
RMSprop_1/rho/initial_valueConst*
valueB
 *fff?*
dtype0*
_output_shapes
: 
q
RMSprop_1/rho
VariableV2*
shape: *
dtype0*
	container *
shared_name *
_output_shapes
: 
Ж
RMSprop_1/rho/AssignAssignRMSprop_1/rhoRMSprop_1/rho/initial_value*
T0*
validate_shape(*
use_locking(* 
_class
loc:@RMSprop_1/rho*
_output_shapes
: 
p
RMSprop_1/rho/readIdentityRMSprop_1/rho*
T0* 
_class
loc:@RMSprop_1/rho*
_output_shapes
: 
b
RMSprop_1/decay/initial_valueConst*
valueB
 *    *
dtype0*
_output_shapes
: 
s
RMSprop_1/decay
VariableV2*
shape: *
dtype0*
	container *
shared_name *
_output_shapes
: 
О
RMSprop_1/decay/AssignAssignRMSprop_1/decayRMSprop_1/decay/initial_value*
T0*
validate_shape(*
use_locking(*"
_class
loc:@RMSprop_1/decay*
_output_shapes
: 
v
RMSprop_1/decay/readIdentityRMSprop_1/decay*
T0*"
_class
loc:@RMSprop_1/decay*
_output_shapes
: 
d
"RMSprop_1/iterations/initial_valueConst*
value	B	 R *
dtype0	*
_output_shapes
: 
x
RMSprop_1/iterations
VariableV2*
shape: *
dtype0	*
	container *
shared_name *
_output_shapes
: 
в
RMSprop_1/iterations/AssignAssignRMSprop_1/iterations"RMSprop_1/iterations/initial_value*
T0	*
validate_shape(*
use_locking(*'
_class
loc:@RMSprop_1/iterations*
_output_shapes
: 

RMSprop_1/iterations/readIdentityRMSprop_1/iterations*
T0	*'
_class
loc:@RMSprop_1/iterations*
_output_shapes
: 

dense_6_targetPlaceholder*
dtype0*%
shape:џџџџџџџџџџџџџџџџџџ*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
q
dense_6_sample_weightsPlaceholder*
dtype0*
shape:џџџџџџџџџ*#
_output_shapes
:џџџџџџџџџ
z
loss_1/dense_6_loss/subSubdense_6/BiasAdddense_6_target*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
x
loss_1/dense_6_loss/SquareSquareloss_1/dense_6_loss/sub*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
u
*loss_1/dense_6_loss/Mean/reduction_indicesConst*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 
Г
loss_1/dense_6_loss/MeanMeanloss_1/dense_6_loss/Square*loss_1/dense_6_loss/Mean/reduction_indices*
	keep_dims( *
T0*

Tidx0*#
_output_shapes
:џџџџџџџџџ
o
,loss_1/dense_6_loss/Mean_1/reduction_indicesConst*
valueB *
dtype0*
_output_shapes
: 
Е
loss_1/dense_6_loss/Mean_1Meanloss_1/dense_6_loss/Mean,loss_1/dense_6_loss/Mean_1/reduction_indices*
	keep_dims( *
T0*

Tidx0*#
_output_shapes
:џџџџџџџџџ

loss_1/dense_6_loss/mulMulloss_1/dense_6_loss/Mean_1dense_6_sample_weights*
T0*#
_output_shapes
:џџџџџџџџџ
c
loss_1/dense_6_loss/NotEqual/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 

loss_1/dense_6_loss/NotEqualNotEqualdense_6_sample_weightsloss_1/dense_6_loss/NotEqual/y*
T0*#
_output_shapes
:џџџџџџџџџ
{
loss_1/dense_6_loss/CastCastloss_1/dense_6_loss/NotEqual*

SrcT0
*

DstT0*#
_output_shapes
:џџџџџџџџџ
c
loss_1/dense_6_loss/ConstConst*
valueB: *
dtype0*
_output_shapes
:

loss_1/dense_6_loss/Mean_2Meanloss_1/dense_6_loss/Castloss_1/dense_6_loss/Const*
	keep_dims( *
T0*

Tidx0*
_output_shapes
: 

loss_1/dense_6_loss/truedivRealDivloss_1/dense_6_loss/mulloss_1/dense_6_loss/Mean_2*
T0*#
_output_shapes
:џџџџџџџџџ
e
loss_1/dense_6_loss/Const_1Const*
valueB: *
dtype0*
_output_shapes
:

loss_1/dense_6_loss/Mean_3Meanloss_1/dense_6_loss/truedivloss_1/dense_6_loss/Const_1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
: 
Q
loss_1/mul/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
\

loss_1/mulMulloss_1/mul/xloss_1/dense_6_loss/Mean_3*
T0*
_output_shapes
: 

!metrics_1/mean_absolute_error/subSubdense_6/BiasAdddense_6_target*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ

!metrics_1/mean_absolute_error/AbsAbs!metrics_1/mean_absolute_error/sub*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ

4metrics_1/mean_absolute_error/Mean/reduction_indicesConst*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 
Ю
"metrics_1/mean_absolute_error/MeanMean!metrics_1/mean_absolute_error/Abs4metrics_1/mean_absolute_error/Mean/reduction_indices*
	keep_dims( *
T0*

Tidx0*#
_output_shapes
:џџџџџџџџџ
m
#metrics_1/mean_absolute_error/ConstConst*
valueB: *
dtype0*
_output_shapes
:
Г
$metrics_1/mean_absolute_error/Mean_1Mean"metrics_1/mean_absolute_error/Mean#metrics_1/mean_absolute_error/Const*
	keep_dims( *
T0*

Tidx0*
_output_shapes
: 

"training_1/RMSprop/gradients/ShapeConst*
valueB *
dtype0*
_class
loc:@loss_1/mul*
_output_shapes
: 

&training_1/RMSprop/gradients/grad_ys_0Const*
valueB
 *  ?*
dtype0*
_class
loc:@loss_1/mul*
_output_shapes
: 
Е
!training_1/RMSprop/gradients/FillFill"training_1/RMSprop/gradients/Shape&training_1/RMSprop/gradients/grad_ys_0*
T0*
_class
loc:@loss_1/mul*
_output_shapes
: 
Ж
0training_1/RMSprop/gradients/loss_1/mul_grad/MulMul!training_1/RMSprop/gradients/Fillloss_1/dense_6_loss/Mean_3*
T0*
_class
loc:@loss_1/mul*
_output_shapes
: 
Њ
2training_1/RMSprop/gradients/loss_1/mul_grad/Mul_1Mul!training_1/RMSprop/gradients/Fillloss_1/mul/x*
T0*
_class
loc:@loss_1/mul*
_output_shapes
: 
У
Jtraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_3_grad/Reshape/shapeConst*
valueB:*
dtype0*-
_class#
!loc:@loss_1/dense_6_loss/Mean_3*
_output_shapes
:
Б
Dtraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_3_grad/ReshapeReshape2training_1/RMSprop/gradients/loss_1/mul_grad/Mul_1Jtraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_3_grad/Reshape/shape*
T0*
Tshape0*-
_class#
!loc:@loss_1/dense_6_loss/Mean_3*
_output_shapes
:
Ь
Btraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_3_grad/ShapeShapeloss_1/dense_6_loss/truediv*
T0*
out_type0*-
_class#
!loc:@loss_1/dense_6_loss/Mean_3*
_output_shapes
:
Т
Atraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_3_grad/TileTileDtraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_3_grad/ReshapeBtraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_3_grad/Shape*
T0*

Tmultiples0*-
_class#
!loc:@loss_1/dense_6_loss/Mean_3*#
_output_shapes
:џџџџџџџџџ
Ю
Dtraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_3_grad/Shape_1Shapeloss_1/dense_6_loss/truediv*
T0*
out_type0*-
_class#
!loc:@loss_1/dense_6_loss/Mean_3*
_output_shapes
:
Ж
Dtraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_3_grad/Shape_2Const*
valueB *
dtype0*-
_class#
!loc:@loss_1/dense_6_loss/Mean_3*
_output_shapes
: 
Л
Btraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_3_grad/ConstConst*
valueB: *
dtype0*-
_class#
!loc:@loss_1/dense_6_loss/Mean_3*
_output_shapes
:
Р
Atraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_3_grad/ProdProdDtraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_3_grad/Shape_1Btraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_3_grad/Const*
	keep_dims( *
T0*

Tidx0*-
_class#
!loc:@loss_1/dense_6_loss/Mean_3*
_output_shapes
: 
Н
Dtraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_3_grad/Const_1Const*
valueB: *
dtype0*-
_class#
!loc:@loss_1/dense_6_loss/Mean_3*
_output_shapes
:
Ф
Ctraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_3_grad/Prod_1ProdDtraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_3_grad/Shape_2Dtraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_3_grad/Const_1*
	keep_dims( *
T0*

Tidx0*-
_class#
!loc:@loss_1/dense_6_loss/Mean_3*
_output_shapes
: 
З
Ftraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_3_grad/Maximum/yConst*
value	B :*
dtype0*-
_class#
!loc:@loss_1/dense_6_loss/Mean_3*
_output_shapes
: 
Ќ
Dtraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_3_grad/MaximumMaximumCtraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_3_grad/Prod_1Ftraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_3_grad/Maximum/y*
T0*-
_class#
!loc:@loss_1/dense_6_loss/Mean_3*
_output_shapes
: 
Њ
Etraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_3_grad/floordivFloorDivAtraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_3_grad/ProdDtraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_3_grad/Maximum*
T0*-
_class#
!loc:@loss_1/dense_6_loss/Mean_3*
_output_shapes
: 
я
Atraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_3_grad/CastCastEtraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_3_grad/floordiv*

SrcT0*

DstT0*-
_class#
!loc:@loss_1/dense_6_loss/Mean_3*
_output_shapes
: 
В
Dtraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_3_grad/truedivRealDivAtraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_3_grad/TileAtraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_3_grad/Cast*
T0*-
_class#
!loc:@loss_1/dense_6_loss/Mean_3*#
_output_shapes
:џџџџџџџџџ
Ъ
Ctraining_1/RMSprop/gradients/loss_1/dense_6_loss/truediv_grad/ShapeShapeloss_1/dense_6_loss/mul*
T0*
out_type0*.
_class$
" loc:@loss_1/dense_6_loss/truediv*
_output_shapes
:
И
Etraining_1/RMSprop/gradients/loss_1/dense_6_loss/truediv_grad/Shape_1Const*
valueB *
dtype0*.
_class$
" loc:@loss_1/dense_6_loss/truediv*
_output_shapes
: 
х
Straining_1/RMSprop/gradients/loss_1/dense_6_loss/truediv_grad/BroadcastGradientArgsBroadcastGradientArgsCtraining_1/RMSprop/gradients/loss_1/dense_6_loss/truediv_grad/ShapeEtraining_1/RMSprop/gradients/loss_1/dense_6_loss/truediv_grad/Shape_1*
T0*.
_class$
" loc:@loss_1/dense_6_loss/truediv*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ

Etraining_1/RMSprop/gradients/loss_1/dense_6_loss/truediv_grad/RealDivRealDivDtraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_3_grad/truedivloss_1/dense_6_loss/Mean_2*
T0*.
_class$
" loc:@loss_1/dense_6_loss/truediv*#
_output_shapes
:џџџџџџџџџ
д
Atraining_1/RMSprop/gradients/loss_1/dense_6_loss/truediv_grad/SumSumEtraining_1/RMSprop/gradients/loss_1/dense_6_loss/truediv_grad/RealDivStraining_1/RMSprop/gradients/loss_1/dense_6_loss/truediv_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*.
_class$
" loc:@loss_1/dense_6_loss/truediv*
_output_shapes
:
Ф
Etraining_1/RMSprop/gradients/loss_1/dense_6_loss/truediv_grad/ReshapeReshapeAtraining_1/RMSprop/gradients/loss_1/dense_6_loss/truediv_grad/SumCtraining_1/RMSprop/gradients/loss_1/dense_6_loss/truediv_grad/Shape*
T0*
Tshape0*.
_class$
" loc:@loss_1/dense_6_loss/truediv*#
_output_shapes
:џџџџџџџџџ
П
Atraining_1/RMSprop/gradients/loss_1/dense_6_loss/truediv_grad/NegNegloss_1/dense_6_loss/mul*
T0*.
_class$
" loc:@loss_1/dense_6_loss/truediv*#
_output_shapes
:џџџџџџџџџ

Gtraining_1/RMSprop/gradients/loss_1/dense_6_loss/truediv_grad/RealDiv_1RealDivAtraining_1/RMSprop/gradients/loss_1/dense_6_loss/truediv_grad/Negloss_1/dense_6_loss/Mean_2*
T0*.
_class$
" loc:@loss_1/dense_6_loss/truediv*#
_output_shapes
:џџџџџџџџџ

Gtraining_1/RMSprop/gradients/loss_1/dense_6_loss/truediv_grad/RealDiv_2RealDivGtraining_1/RMSprop/gradients/loss_1/dense_6_loss/truediv_grad/RealDiv_1loss_1/dense_6_loss/Mean_2*
T0*.
_class$
" loc:@loss_1/dense_6_loss/truediv*#
_output_shapes
:џџџџџџџџџ
Е
Atraining_1/RMSprop/gradients/loss_1/dense_6_loss/truediv_grad/mulMulDtraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_3_grad/truedivGtraining_1/RMSprop/gradients/loss_1/dense_6_loss/truediv_grad/RealDiv_2*
T0*.
_class$
" loc:@loss_1/dense_6_loss/truediv*#
_output_shapes
:џџџџџџџџџ
д
Ctraining_1/RMSprop/gradients/loss_1/dense_6_loss/truediv_grad/Sum_1SumAtraining_1/RMSprop/gradients/loss_1/dense_6_loss/truediv_grad/mulUtraining_1/RMSprop/gradients/loss_1/dense_6_loss/truediv_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*.
_class$
" loc:@loss_1/dense_6_loss/truediv*
_output_shapes
:
Н
Gtraining_1/RMSprop/gradients/loss_1/dense_6_loss/truediv_grad/Reshape_1ReshapeCtraining_1/RMSprop/gradients/loss_1/dense_6_loss/truediv_grad/Sum_1Etraining_1/RMSprop/gradients/loss_1/dense_6_loss/truediv_grad/Shape_1*
T0*
Tshape0*.
_class$
" loc:@loss_1/dense_6_loss/truediv*
_output_shapes
: 
Х
?training_1/RMSprop/gradients/loss_1/dense_6_loss/mul_grad/ShapeShapeloss_1/dense_6_loss/Mean_1*
T0*
out_type0**
_class 
loc:@loss_1/dense_6_loss/mul*
_output_shapes
:
У
Atraining_1/RMSprop/gradients/loss_1/dense_6_loss/mul_grad/Shape_1Shapedense_6_sample_weights*
T0*
out_type0**
_class 
loc:@loss_1/dense_6_loss/mul*
_output_shapes
:
е
Otraining_1/RMSprop/gradients/loss_1/dense_6_loss/mul_grad/BroadcastGradientArgsBroadcastGradientArgs?training_1/RMSprop/gradients/loss_1/dense_6_loss/mul_grad/ShapeAtraining_1/RMSprop/gradients/loss_1/dense_6_loss/mul_grad/Shape_1*
T0**
_class 
loc:@loss_1/dense_6_loss/mul*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
§
=training_1/RMSprop/gradients/loss_1/dense_6_loss/mul_grad/mulMulEtraining_1/RMSprop/gradients/loss_1/dense_6_loss/truediv_grad/Reshapedense_6_sample_weights*
T0**
_class 
loc:@loss_1/dense_6_loss/mul*#
_output_shapes
:џџџџџџџџџ
Р
=training_1/RMSprop/gradients/loss_1/dense_6_loss/mul_grad/SumSum=training_1/RMSprop/gradients/loss_1/dense_6_loss/mul_grad/mulOtraining_1/RMSprop/gradients/loss_1/dense_6_loss/mul_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0**
_class 
loc:@loss_1/dense_6_loss/mul*
_output_shapes
:
Д
Atraining_1/RMSprop/gradients/loss_1/dense_6_loss/mul_grad/ReshapeReshape=training_1/RMSprop/gradients/loss_1/dense_6_loss/mul_grad/Sum?training_1/RMSprop/gradients/loss_1/dense_6_loss/mul_grad/Shape*
T0*
Tshape0**
_class 
loc:@loss_1/dense_6_loss/mul*#
_output_shapes
:џџџџџџџџџ

?training_1/RMSprop/gradients/loss_1/dense_6_loss/mul_grad/mul_1Mulloss_1/dense_6_loss/Mean_1Etraining_1/RMSprop/gradients/loss_1/dense_6_loss/truediv_grad/Reshape*
T0**
_class 
loc:@loss_1/dense_6_loss/mul*#
_output_shapes
:џџџџџџџџџ
Ц
?training_1/RMSprop/gradients/loss_1/dense_6_loss/mul_grad/Sum_1Sum?training_1/RMSprop/gradients/loss_1/dense_6_loss/mul_grad/mul_1Qtraining_1/RMSprop/gradients/loss_1/dense_6_loss/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0**
_class 
loc:@loss_1/dense_6_loss/mul*
_output_shapes
:
К
Ctraining_1/RMSprop/gradients/loss_1/dense_6_loss/mul_grad/Reshape_1Reshape?training_1/RMSprop/gradients/loss_1/dense_6_loss/mul_grad/Sum_1Atraining_1/RMSprop/gradients/loss_1/dense_6_loss/mul_grad/Shape_1*
T0*
Tshape0**
_class 
loc:@loss_1/dense_6_loss/mul*#
_output_shapes
:џџџџџџџџџ
Щ
Btraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_1_grad/ShapeShapeloss_1/dense_6_loss/Mean*
T0*
out_type0*-
_class#
!loc:@loss_1/dense_6_loss/Mean_1*
_output_shapes
:
В
Atraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_1_grad/SizeConst*
value	B :*
dtype0*-
_class#
!loc:@loss_1/dense_6_loss/Mean_1*
_output_shapes
: 

@training_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_1_grad/addAdd,loss_1/dense_6_loss/Mean_1/reduction_indicesAtraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_1_grad/Size*
T0*-
_class#
!loc:@loss_1/dense_6_loss/Mean_1*
_output_shapes
: 
Ѓ
@training_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_1_grad/modFloorMod@training_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_1_grad/addAtraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_1_grad/Size*
T0*-
_class#
!loc:@loss_1/dense_6_loss/Mean_1*
_output_shapes
: 
Н
Dtraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_1_grad/Shape_1Const*
valueB: *
dtype0*-
_class#
!loc:@loss_1/dense_6_loss/Mean_1*
_output_shapes
:
Й
Htraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_1_grad/range/startConst*
value	B : *
dtype0*-
_class#
!loc:@loss_1/dense_6_loss/Mean_1*
_output_shapes
: 
Й
Htraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_1_grad/range/deltaConst*
value	B :*
dtype0*-
_class#
!loc:@loss_1/dense_6_loss/Mean_1*
_output_shapes
: 
љ
Btraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_1_grad/rangeRangeHtraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_1_grad/range/startAtraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_1_grad/SizeHtraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_1_grad/range/delta*

Tidx0*-
_class#
!loc:@loss_1/dense_6_loss/Mean_1*
_output_shapes
:
И
Gtraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_1_grad/Fill/valueConst*
value	B :*
dtype0*-
_class#
!loc:@loss_1/dense_6_loss/Mean_1*
_output_shapes
: 
Њ
Atraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_1_grad/FillFillDtraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_1_grad/Shape_1Gtraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_1_grad/Fill/value*
T0*-
_class#
!loc:@loss_1/dense_6_loss/Mean_1*
_output_shapes
: 
Ю
Jtraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_1_grad/DynamicStitchDynamicStitchBtraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_1_grad/range@training_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_1_grad/modBtraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_1_grad/ShapeAtraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_1_grad/Fill*
N*
T0*-
_class#
!loc:@loss_1/dense_6_loss/Mean_1*#
_output_shapes
:џџџџџџџџџ
З
Ftraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_1_grad/Maximum/yConst*
value	B :*
dtype0*-
_class#
!loc:@loss_1/dense_6_loss/Mean_1*
_output_shapes
: 
Р
Dtraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_1_grad/MaximumMaximumJtraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_1_grad/DynamicStitchFtraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_1_grad/Maximum/y*
T0*-
_class#
!loc:@loss_1/dense_6_loss/Mean_1*#
_output_shapes
:џџџџџџџџџ
И
Etraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_1_grad/floordivFloorDivBtraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_1_grad/ShapeDtraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_1_grad/Maximum*
T0*-
_class#
!loc:@loss_1/dense_6_loss/Mean_1*#
_output_shapes
:џџџџџџџџџ
О
Dtraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_1_grad/ReshapeReshapeAtraining_1/RMSprop/gradients/loss_1/dense_6_loss/mul_grad/ReshapeJtraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_1_grad/DynamicStitch*
T0*
Tshape0*-
_class#
!loc:@loss_1/dense_6_loss/Mean_1*
_output_shapes
:
К
Atraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_1_grad/TileTileDtraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_1_grad/ReshapeEtraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_1_grad/floordiv*
T0*

Tmultiples0*-
_class#
!loc:@loss_1/dense_6_loss/Mean_1*
_output_shapes
:
Ы
Dtraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_1_grad/Shape_2Shapeloss_1/dense_6_loss/Mean*
T0*
out_type0*-
_class#
!loc:@loss_1/dense_6_loss/Mean_1*
_output_shapes
:
Э
Dtraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_1_grad/Shape_3Shapeloss_1/dense_6_loss/Mean_1*
T0*
out_type0*-
_class#
!loc:@loss_1/dense_6_loss/Mean_1*
_output_shapes
:
Л
Btraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_1_grad/ConstConst*
valueB: *
dtype0*-
_class#
!loc:@loss_1/dense_6_loss/Mean_1*
_output_shapes
:
Р
Atraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_1_grad/ProdProdDtraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_1_grad/Shape_2Btraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_1_grad/Const*
	keep_dims( *
T0*

Tidx0*-
_class#
!loc:@loss_1/dense_6_loss/Mean_1*
_output_shapes
: 
Н
Dtraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_1_grad/Const_1Const*
valueB: *
dtype0*-
_class#
!loc:@loss_1/dense_6_loss/Mean_1*
_output_shapes
:
Ф
Ctraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_1_grad/Prod_1ProdDtraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_1_grad/Shape_3Dtraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_1_grad/Const_1*
	keep_dims( *
T0*

Tidx0*-
_class#
!loc:@loss_1/dense_6_loss/Mean_1*
_output_shapes
: 
Й
Htraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_1_grad/Maximum_1/yConst*
value	B :*
dtype0*-
_class#
!loc:@loss_1/dense_6_loss/Mean_1*
_output_shapes
: 
А
Ftraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_1_grad/Maximum_1MaximumCtraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_1_grad/Prod_1Htraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_1_grad/Maximum_1/y*
T0*-
_class#
!loc:@loss_1/dense_6_loss/Mean_1*
_output_shapes
: 
Ў
Gtraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_1_grad/floordiv_1FloorDivAtraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_1_grad/ProdFtraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_1_grad/Maximum_1*
T0*-
_class#
!loc:@loss_1/dense_6_loss/Mean_1*
_output_shapes
: 
ё
Atraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_1_grad/CastCastGtraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_1_grad/floordiv_1*

SrcT0*

DstT0*-
_class#
!loc:@loss_1/dense_6_loss/Mean_1*
_output_shapes
: 
В
Dtraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_1_grad/truedivRealDivAtraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_1_grad/TileAtraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_1_grad/Cast*
T0*-
_class#
!loc:@loss_1/dense_6_loss/Mean_1*#
_output_shapes
:џџџџџџџџџ
Ч
@training_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_grad/ShapeShapeloss_1/dense_6_loss/Square*
T0*
out_type0*+
_class!
loc:@loss_1/dense_6_loss/Mean*
_output_shapes
:
Ў
?training_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_grad/SizeConst*
value	B :*
dtype0*+
_class!
loc:@loss_1/dense_6_loss/Mean*
_output_shapes
: 

>training_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_grad/addAdd*loss_1/dense_6_loss/Mean/reduction_indices?training_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_grad/Size*
T0*+
_class!
loc:@loss_1/dense_6_loss/Mean*
_output_shapes
: 

>training_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_grad/modFloorMod>training_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_grad/add?training_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_grad/Size*
T0*+
_class!
loc:@loss_1/dense_6_loss/Mean*
_output_shapes
: 
В
Btraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_grad/Shape_1Const*
valueB *
dtype0*+
_class!
loc:@loss_1/dense_6_loss/Mean*
_output_shapes
: 
Е
Ftraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_grad/range/startConst*
value	B : *
dtype0*+
_class!
loc:@loss_1/dense_6_loss/Mean*
_output_shapes
: 
Е
Ftraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_grad/range/deltaConst*
value	B :*
dtype0*+
_class!
loc:@loss_1/dense_6_loss/Mean*
_output_shapes
: 
я
@training_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_grad/rangeRangeFtraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_grad/range/start?training_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_grad/SizeFtraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_grad/range/delta*

Tidx0*+
_class!
loc:@loss_1/dense_6_loss/Mean*
_output_shapes
:
Д
Etraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_grad/Fill/valueConst*
value	B :*
dtype0*+
_class!
loc:@loss_1/dense_6_loss/Mean*
_output_shapes
: 
 
?training_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_grad/FillFillBtraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_grad/Shape_1Etraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_grad/Fill/value*
T0*+
_class!
loc:@loss_1/dense_6_loss/Mean*
_output_shapes
: 
Т
Htraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_grad/DynamicStitchDynamicStitch@training_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_grad/range>training_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_grad/mod@training_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_grad/Shape?training_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_grad/Fill*
N*
T0*+
_class!
loc:@loss_1/dense_6_loss/Mean*#
_output_shapes
:џџџџџџџџџ
Г
Dtraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_grad/Maximum/yConst*
value	B :*
dtype0*+
_class!
loc:@loss_1/dense_6_loss/Mean*
_output_shapes
: 
И
Btraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_grad/MaximumMaximumHtraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_grad/DynamicStitchDtraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_grad/Maximum/y*
T0*+
_class!
loc:@loss_1/dense_6_loss/Mean*#
_output_shapes
:џџџџџџџџџ
Ї
Ctraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_grad/floordivFloorDiv@training_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_grad/ShapeBtraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_grad/Maximum*
T0*+
_class!
loc:@loss_1/dense_6_loss/Mean*
_output_shapes
:
Л
Btraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_grad/ReshapeReshapeDtraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_1_grad/truedivHtraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_grad/DynamicStitch*
T0*
Tshape0*+
_class!
loc:@loss_1/dense_6_loss/Mean*
_output_shapes
:
Ъ
?training_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_grad/TileTileBtraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_grad/ReshapeCtraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_grad/floordiv*
T0*

Tmultiples0*+
_class!
loc:@loss_1/dense_6_loss/Mean*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
Щ
Btraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_grad/Shape_2Shapeloss_1/dense_6_loss/Square*
T0*
out_type0*+
_class!
loc:@loss_1/dense_6_loss/Mean*
_output_shapes
:
Ч
Btraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_grad/Shape_3Shapeloss_1/dense_6_loss/Mean*
T0*
out_type0*+
_class!
loc:@loss_1/dense_6_loss/Mean*
_output_shapes
:
З
@training_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_grad/ConstConst*
valueB: *
dtype0*+
_class!
loc:@loss_1/dense_6_loss/Mean*
_output_shapes
:
И
?training_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_grad/ProdProdBtraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_grad/Shape_2@training_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_grad/Const*
	keep_dims( *
T0*

Tidx0*+
_class!
loc:@loss_1/dense_6_loss/Mean*
_output_shapes
: 
Й
Btraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_grad/Const_1Const*
valueB: *
dtype0*+
_class!
loc:@loss_1/dense_6_loss/Mean*
_output_shapes
:
М
Atraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_grad/Prod_1ProdBtraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_grad/Shape_3Btraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_grad/Const_1*
	keep_dims( *
T0*

Tidx0*+
_class!
loc:@loss_1/dense_6_loss/Mean*
_output_shapes
: 
Е
Ftraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_grad/Maximum_1/yConst*
value	B :*
dtype0*+
_class!
loc:@loss_1/dense_6_loss/Mean*
_output_shapes
: 
Ј
Dtraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_grad/Maximum_1MaximumAtraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_grad/Prod_1Ftraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_grad/Maximum_1/y*
T0*+
_class!
loc:@loss_1/dense_6_loss/Mean*
_output_shapes
: 
І
Etraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_grad/floordiv_1FloorDiv?training_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_grad/ProdDtraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_grad/Maximum_1*
T0*+
_class!
loc:@loss_1/dense_6_loss/Mean*
_output_shapes
: 
ы
?training_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_grad/CastCastEtraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_grad/floordiv_1*

SrcT0*

DstT0*+
_class!
loc:@loss_1/dense_6_loss/Mean*
_output_shapes
: 
З
Btraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_grad/truedivRealDiv?training_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_grad/Tile?training_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_grad/Cast*
T0*+
_class!
loc:@loss_1/dense_6_loss/Mean*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
ћ
Btraining_1/RMSprop/gradients/loss_1/dense_6_loss/Square_grad/mul/xConstC^training_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_grad/truediv*
valueB
 *   @*
dtype0*-
_class#
!loc:@loss_1/dense_6_loss/Square*
_output_shapes
: 

@training_1/RMSprop/gradients/loss_1/dense_6_loss/Square_grad/mulMulBtraining_1/RMSprop/gradients/loss_1/dense_6_loss/Square_grad/mul/xloss_1/dense_6_loss/sub*
T0*-
_class#
!loc:@loss_1/dense_6_loss/Square*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
Й
Btraining_1/RMSprop/gradients/loss_1/dense_6_loss/Square_grad/mul_1MulBtraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_grad/truediv@training_1/RMSprop/gradients/loss_1/dense_6_loss/Square_grad/mul*
T0*-
_class#
!loc:@loss_1/dense_6_loss/Square*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
К
?training_1/RMSprop/gradients/loss_1/dense_6_loss/sub_grad/ShapeShapedense_6/BiasAdd*
T0*
out_type0**
_class 
loc:@loss_1/dense_6_loss/sub*
_output_shapes
:
Л
Atraining_1/RMSprop/gradients/loss_1/dense_6_loss/sub_grad/Shape_1Shapedense_6_target*
T0*
out_type0**
_class 
loc:@loss_1/dense_6_loss/sub*
_output_shapes
:
е
Otraining_1/RMSprop/gradients/loss_1/dense_6_loss/sub_grad/BroadcastGradientArgsBroadcastGradientArgs?training_1/RMSprop/gradients/loss_1/dense_6_loss/sub_grad/ShapeAtraining_1/RMSprop/gradients/loss_1/dense_6_loss/sub_grad/Shape_1*
T0**
_class 
loc:@loss_1/dense_6_loss/sub*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Х
=training_1/RMSprop/gradients/loss_1/dense_6_loss/sub_grad/SumSumBtraining_1/RMSprop/gradients/loss_1/dense_6_loss/Square_grad/mul_1Otraining_1/RMSprop/gradients/loss_1/dense_6_loss/sub_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0**
_class 
loc:@loss_1/dense_6_loss/sub*
_output_shapes
:
И
Atraining_1/RMSprop/gradients/loss_1/dense_6_loss/sub_grad/ReshapeReshape=training_1/RMSprop/gradients/loss_1/dense_6_loss/sub_grad/Sum?training_1/RMSprop/gradients/loss_1/dense_6_loss/sub_grad/Shape*
T0*
Tshape0**
_class 
loc:@loss_1/dense_6_loss/sub*'
_output_shapes
:џџџџџџџџџ
Щ
?training_1/RMSprop/gradients/loss_1/dense_6_loss/sub_grad/Sum_1SumBtraining_1/RMSprop/gradients/loss_1/dense_6_loss/Square_grad/mul_1Qtraining_1/RMSprop/gradients/loss_1/dense_6_loss/sub_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0**
_class 
loc:@loss_1/dense_6_loss/sub*
_output_shapes
:
д
=training_1/RMSprop/gradients/loss_1/dense_6_loss/sub_grad/NegNeg?training_1/RMSprop/gradients/loss_1/dense_6_loss/sub_grad/Sum_1*
T0**
_class 
loc:@loss_1/dense_6_loss/sub*
_output_shapes
:
Х
Ctraining_1/RMSprop/gradients/loss_1/dense_6_loss/sub_grad/Reshape_1Reshape=training_1/RMSprop/gradients/loss_1/dense_6_loss/sub_grad/NegAtraining_1/RMSprop/gradients/loss_1/dense_6_loss/sub_grad/Shape_1*
T0*
Tshape0**
_class 
loc:@loss_1/dense_6_loss/sub*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
я
=training_1/RMSprop/gradients/dense_6/BiasAdd_grad/BiasAddGradBiasAddGradAtraining_1/RMSprop/gradients/loss_1/dense_6_loss/sub_grad/Reshape*
T0*
data_formatNHWC*"
_class
loc:@dense_6/BiasAdd*
_output_shapes
:

7training_1/RMSprop/gradients/dense_6/MatMul_grad/MatMulMatMulAtraining_1/RMSprop/gradients/loss_1/dense_6_loss/sub_grad/Reshapedense_6/kernel/read*
transpose_a( *
transpose_b(*
T0*!
_class
loc:@dense_6/MatMul*'
_output_shapes
:џџџџџџџџџ@

9training_1/RMSprop/gradients/dense_6/MatMul_grad/MatMul_1MatMuldense_5/ReluAtraining_1/RMSprop/gradients/loss_1/dense_6_loss/sub_grad/Reshape*
transpose_a(*
transpose_b( *
T0*!
_class
loc:@dense_6/MatMul*
_output_shapes

:@
н
7training_1/RMSprop/gradients/dense_5/Relu_grad/ReluGradReluGrad7training_1/RMSprop/gradients/dense_6/MatMul_grad/MatMuldense_5/Relu*
T0*
_class
loc:@dense_5/Relu*'
_output_shapes
:џџџџџџџџџ@
х
=training_1/RMSprop/gradients/dense_5/BiasAdd_grad/BiasAddGradBiasAddGrad7training_1/RMSprop/gradients/dense_5/Relu_grad/ReluGrad*
T0*
data_formatNHWC*"
_class
loc:@dense_5/BiasAdd*
_output_shapes
:@

7training_1/RMSprop/gradients/dense_5/MatMul_grad/MatMulMatMul7training_1/RMSprop/gradients/dense_5/Relu_grad/ReluGraddense_5/kernel/read*
transpose_a( *
transpose_b(*
T0*!
_class
loc:@dense_5/MatMul*'
_output_shapes
:џџџџџџџџџ@
ќ
9training_1/RMSprop/gradients/dense_5/MatMul_grad/MatMul_1MatMuldense_4/Relu7training_1/RMSprop/gradients/dense_5/Relu_grad/ReluGrad*
transpose_a(*
transpose_b( *
T0*!
_class
loc:@dense_5/MatMul*
_output_shapes

:@@
н
7training_1/RMSprop/gradients/dense_4/Relu_grad/ReluGradReluGrad7training_1/RMSprop/gradients/dense_5/MatMul_grad/MatMuldense_4/Relu*
T0*
_class
loc:@dense_4/Relu*'
_output_shapes
:џџџџџџџџџ@
х
=training_1/RMSprop/gradients/dense_4/BiasAdd_grad/BiasAddGradBiasAddGrad7training_1/RMSprop/gradients/dense_4/Relu_grad/ReluGrad*
T0*
data_formatNHWC*"
_class
loc:@dense_4/BiasAdd*
_output_shapes
:@

7training_1/RMSprop/gradients/dense_4/MatMul_grad/MatMulMatMul7training_1/RMSprop/gradients/dense_4/Relu_grad/ReluGraddense_4/kernel/read*
transpose_a( *
transpose_b(*
T0*!
_class
loc:@dense_4/MatMul*'
_output_shapes
:џџџџџџџџџ
§
9training_1/RMSprop/gradients/dense_4/MatMul_grad/MatMul_1MatMuldense_4_input7training_1/RMSprop/gradients/dense_4/Relu_grad/ReluGrad*
transpose_a(*
transpose_b( *
T0*!
_class
loc:@dense_4/MatMul*
_output_shapes

:@
m
training_1/RMSprop/ConstConst*
valueB@*    *
dtype0*
_output_shapes

:@

training_1/RMSprop/Variable
VariableV2*
shape
:@*
dtype0*
	container *
shared_name *
_output_shapes

:@
х
"training_1/RMSprop/Variable/AssignAssigntraining_1/RMSprop/Variabletraining_1/RMSprop/Const*
T0*
validate_shape(*
use_locking(*.
_class$
" loc:@training_1/RMSprop/Variable*
_output_shapes

:@
Ђ
 training_1/RMSprop/Variable/readIdentitytraining_1/RMSprop/Variable*
T0*.
_class$
" loc:@training_1/RMSprop/Variable*
_output_shapes

:@
g
training_1/RMSprop/Const_1Const*
valueB@*    *
dtype0*
_output_shapes
:@

training_1/RMSprop/Variable_1
VariableV2*
shape:@*
dtype0*
	container *
shared_name *
_output_shapes
:@
щ
$training_1/RMSprop/Variable_1/AssignAssigntraining_1/RMSprop/Variable_1training_1/RMSprop/Const_1*
T0*
validate_shape(*
use_locking(*0
_class&
$"loc:@training_1/RMSprop/Variable_1*
_output_shapes
:@
Є
"training_1/RMSprop/Variable_1/readIdentitytraining_1/RMSprop/Variable_1*
T0*0
_class&
$"loc:@training_1/RMSprop/Variable_1*
_output_shapes
:@
o
training_1/RMSprop/Const_2Const*
valueB@@*    *
dtype0*
_output_shapes

:@@

training_1/RMSprop/Variable_2
VariableV2*
shape
:@@*
dtype0*
	container *
shared_name *
_output_shapes

:@@
э
$training_1/RMSprop/Variable_2/AssignAssigntraining_1/RMSprop/Variable_2training_1/RMSprop/Const_2*
T0*
validate_shape(*
use_locking(*0
_class&
$"loc:@training_1/RMSprop/Variable_2*
_output_shapes

:@@
Ј
"training_1/RMSprop/Variable_2/readIdentitytraining_1/RMSprop/Variable_2*
T0*0
_class&
$"loc:@training_1/RMSprop/Variable_2*
_output_shapes

:@@
g
training_1/RMSprop/Const_3Const*
valueB@*    *
dtype0*
_output_shapes
:@

training_1/RMSprop/Variable_3
VariableV2*
shape:@*
dtype0*
	container *
shared_name *
_output_shapes
:@
щ
$training_1/RMSprop/Variable_3/AssignAssigntraining_1/RMSprop/Variable_3training_1/RMSprop/Const_3*
T0*
validate_shape(*
use_locking(*0
_class&
$"loc:@training_1/RMSprop/Variable_3*
_output_shapes
:@
Є
"training_1/RMSprop/Variable_3/readIdentitytraining_1/RMSprop/Variable_3*
T0*0
_class&
$"loc:@training_1/RMSprop/Variable_3*
_output_shapes
:@
o
training_1/RMSprop/Const_4Const*
valueB@*    *
dtype0*
_output_shapes

:@

training_1/RMSprop/Variable_4
VariableV2*
shape
:@*
dtype0*
	container *
shared_name *
_output_shapes

:@
э
$training_1/RMSprop/Variable_4/AssignAssigntraining_1/RMSprop/Variable_4training_1/RMSprop/Const_4*
T0*
validate_shape(*
use_locking(*0
_class&
$"loc:@training_1/RMSprop/Variable_4*
_output_shapes

:@
Ј
"training_1/RMSprop/Variable_4/readIdentitytraining_1/RMSprop/Variable_4*
T0*0
_class&
$"loc:@training_1/RMSprop/Variable_4*
_output_shapes

:@
g
training_1/RMSprop/Const_5Const*
valueB*    *
dtype0*
_output_shapes
:

training_1/RMSprop/Variable_5
VariableV2*
shape:*
dtype0*
	container *
shared_name *
_output_shapes
:
щ
$training_1/RMSprop/Variable_5/AssignAssigntraining_1/RMSprop/Variable_5training_1/RMSprop/Const_5*
T0*
validate_shape(*
use_locking(*0
_class&
$"loc:@training_1/RMSprop/Variable_5*
_output_shapes
:
Є
"training_1/RMSprop/Variable_5/readIdentitytraining_1/RMSprop/Variable_5*
T0*0
_class&
$"loc:@training_1/RMSprop/Variable_5*
_output_shapes
:
d
"training_1/RMSprop/AssignAdd/valueConst*
value	B	 R*
dtype0	*
_output_shapes
: 
Р
training_1/RMSprop/AssignAdd	AssignAddRMSprop_1/iterations"training_1/RMSprop/AssignAdd/value*
T0	*
use_locking( *'
_class
loc:@RMSprop_1/iterations*
_output_shapes
: 
|
training_1/RMSprop/mulMulRMSprop_1/rho/read training_1/RMSprop/Variable/read*
T0*
_output_shapes

:@
]
training_1/RMSprop/sub/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
l
training_1/RMSprop/subSubtraining_1/RMSprop/sub/xRMSprop_1/rho/read*
T0*
_output_shapes
: 

training_1/RMSprop/SquareSquare9training_1/RMSprop/gradients/dense_4/MatMul_grad/MatMul_1*
T0*
_output_shapes

:@
{
training_1/RMSprop/mul_1Multraining_1/RMSprop/subtraining_1/RMSprop/Square*
T0*
_output_shapes

:@
x
training_1/RMSprop/addAddtraining_1/RMSprop/multraining_1/RMSprop/mul_1*
T0*
_output_shapes

:@
к
training_1/RMSprop/AssignAssigntraining_1/RMSprop/Variabletraining_1/RMSprop/add*
T0*
validate_shape(*
use_locking(*.
_class$
" loc:@training_1/RMSprop/Variable*
_output_shapes

:@

training_1/RMSprop/mul_2MulRMSprop_1/lr/read9training_1/RMSprop/gradients/dense_4/MatMul_grad/MatMul_1*
T0*
_output_shapes

:@
_
training_1/RMSprop/Const_6Const*
valueB
 *    *
dtype0*
_output_shapes
: 
_
training_1/RMSprop/Const_7Const*
valueB
 *  *
dtype0*
_output_shapes
: 

(training_1/RMSprop/clip_by_value/MinimumMinimumtraining_1/RMSprop/addtraining_1/RMSprop/Const_7*
T0*
_output_shapes

:@

 training_1/RMSprop/clip_by_valueMaximum(training_1/RMSprop/clip_by_value/Minimumtraining_1/RMSprop/Const_6*
T0*
_output_shapes

:@
j
training_1/RMSprop/SqrtSqrt training_1/RMSprop/clip_by_value*
T0*
_output_shapes

:@
_
training_1/RMSprop/add_1/yConst*
valueB
 *wЬ+2*
dtype0*
_output_shapes
: 
}
training_1/RMSprop/add_1Addtraining_1/RMSprop/Sqrttraining_1/RMSprop/add_1/y*
T0*
_output_shapes

:@

training_1/RMSprop/truedivRealDivtraining_1/RMSprop/mul_2training_1/RMSprop/add_1*
T0*
_output_shapes

:@
y
training_1/RMSprop/sub_1Subdense_4/kernel/readtraining_1/RMSprop/truediv*
T0*
_output_shapes

:@
Ф
training_1/RMSprop/Assign_1Assigndense_4/kerneltraining_1/RMSprop/sub_1*
T0*
validate_shape(*
use_locking(*!
_class
loc:@dense_4/kernel*
_output_shapes

:@
|
training_1/RMSprop/mul_3MulRMSprop_1/rho/read"training_1/RMSprop/Variable_1/read*
T0*
_output_shapes
:@
_
training_1/RMSprop/sub_2/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
p
training_1/RMSprop/sub_2Subtraining_1/RMSprop/sub_2/xRMSprop_1/rho/read*
T0*
_output_shapes
: 

training_1/RMSprop/Square_1Square=training_1/RMSprop/gradients/dense_4/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:@
{
training_1/RMSprop/mul_4Multraining_1/RMSprop/sub_2training_1/RMSprop/Square_1*
T0*
_output_shapes
:@
x
training_1/RMSprop/add_2Addtraining_1/RMSprop/mul_3training_1/RMSprop/mul_4*
T0*
_output_shapes
:@
о
training_1/RMSprop/Assign_2Assigntraining_1/RMSprop/Variable_1training_1/RMSprop/add_2*
T0*
validate_shape(*
use_locking(*0
_class&
$"loc:@training_1/RMSprop/Variable_1*
_output_shapes
:@

training_1/RMSprop/mul_5MulRMSprop_1/lr/read=training_1/RMSprop/gradients/dense_4/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:@
_
training_1/RMSprop/Const_8Const*
valueB
 *    *
dtype0*
_output_shapes
: 
_
training_1/RMSprop/Const_9Const*
valueB
 *  *
dtype0*
_output_shapes
: 

*training_1/RMSprop/clip_by_value_1/MinimumMinimumtraining_1/RMSprop/add_2training_1/RMSprop/Const_9*
T0*
_output_shapes
:@

"training_1/RMSprop/clip_by_value_1Maximum*training_1/RMSprop/clip_by_value_1/Minimumtraining_1/RMSprop/Const_8*
T0*
_output_shapes
:@
j
training_1/RMSprop/Sqrt_1Sqrt"training_1/RMSprop/clip_by_value_1*
T0*
_output_shapes
:@
_
training_1/RMSprop/add_3/yConst*
valueB
 *wЬ+2*
dtype0*
_output_shapes
: 
{
training_1/RMSprop/add_3Addtraining_1/RMSprop/Sqrt_1training_1/RMSprop/add_3/y*
T0*
_output_shapes
:@

training_1/RMSprop/truediv_1RealDivtraining_1/RMSprop/mul_5training_1/RMSprop/add_3*
T0*
_output_shapes
:@
u
training_1/RMSprop/sub_3Subdense_4/bias/readtraining_1/RMSprop/truediv_1*
T0*
_output_shapes
:@
М
training_1/RMSprop/Assign_3Assigndense_4/biastraining_1/RMSprop/sub_3*
T0*
validate_shape(*
use_locking(*
_class
loc:@dense_4/bias*
_output_shapes
:@

training_1/RMSprop/mul_6MulRMSprop_1/rho/read"training_1/RMSprop/Variable_2/read*
T0*
_output_shapes

:@@
_
training_1/RMSprop/sub_4/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
p
training_1/RMSprop/sub_4Subtraining_1/RMSprop/sub_4/xRMSprop_1/rho/read*
T0*
_output_shapes
: 

training_1/RMSprop/Square_2Square9training_1/RMSprop/gradients/dense_5/MatMul_grad/MatMul_1*
T0*
_output_shapes

:@@

training_1/RMSprop/mul_7Multraining_1/RMSprop/sub_4training_1/RMSprop/Square_2*
T0*
_output_shapes

:@@
|
training_1/RMSprop/add_4Addtraining_1/RMSprop/mul_6training_1/RMSprop/mul_7*
T0*
_output_shapes

:@@
т
training_1/RMSprop/Assign_4Assigntraining_1/RMSprop/Variable_2training_1/RMSprop/add_4*
T0*
validate_shape(*
use_locking(*0
_class&
$"loc:@training_1/RMSprop/Variable_2*
_output_shapes

:@@

training_1/RMSprop/mul_8MulRMSprop_1/lr/read9training_1/RMSprop/gradients/dense_5/MatMul_grad/MatMul_1*
T0*
_output_shapes

:@@
`
training_1/RMSprop/Const_10Const*
valueB
 *    *
dtype0*
_output_shapes
: 
`
training_1/RMSprop/Const_11Const*
valueB
 *  *
dtype0*
_output_shapes
: 

*training_1/RMSprop/clip_by_value_2/MinimumMinimumtraining_1/RMSprop/add_4training_1/RMSprop/Const_11*
T0*
_output_shapes

:@@

"training_1/RMSprop/clip_by_value_2Maximum*training_1/RMSprop/clip_by_value_2/Minimumtraining_1/RMSprop/Const_10*
T0*
_output_shapes

:@@
n
training_1/RMSprop/Sqrt_2Sqrt"training_1/RMSprop/clip_by_value_2*
T0*
_output_shapes

:@@
_
training_1/RMSprop/add_5/yConst*
valueB
 *wЬ+2*
dtype0*
_output_shapes
: 

training_1/RMSprop/add_5Addtraining_1/RMSprop/Sqrt_2training_1/RMSprop/add_5/y*
T0*
_output_shapes

:@@

training_1/RMSprop/truediv_2RealDivtraining_1/RMSprop/mul_8training_1/RMSprop/add_5*
T0*
_output_shapes

:@@
{
training_1/RMSprop/sub_5Subdense_5/kernel/readtraining_1/RMSprop/truediv_2*
T0*
_output_shapes

:@@
Ф
training_1/RMSprop/Assign_5Assigndense_5/kerneltraining_1/RMSprop/sub_5*
T0*
validate_shape(*
use_locking(*!
_class
loc:@dense_5/kernel*
_output_shapes

:@@
|
training_1/RMSprop/mul_9MulRMSprop_1/rho/read"training_1/RMSprop/Variable_3/read*
T0*
_output_shapes
:@
_
training_1/RMSprop/sub_6/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
p
training_1/RMSprop/sub_6Subtraining_1/RMSprop/sub_6/xRMSprop_1/rho/read*
T0*
_output_shapes
: 

training_1/RMSprop/Square_3Square=training_1/RMSprop/gradients/dense_5/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:@
|
training_1/RMSprop/mul_10Multraining_1/RMSprop/sub_6training_1/RMSprop/Square_3*
T0*
_output_shapes
:@
y
training_1/RMSprop/add_6Addtraining_1/RMSprop/mul_9training_1/RMSprop/mul_10*
T0*
_output_shapes
:@
о
training_1/RMSprop/Assign_6Assigntraining_1/RMSprop/Variable_3training_1/RMSprop/add_6*
T0*
validate_shape(*
use_locking(*0
_class&
$"loc:@training_1/RMSprop/Variable_3*
_output_shapes
:@

training_1/RMSprop/mul_11MulRMSprop_1/lr/read=training_1/RMSprop/gradients/dense_5/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:@
`
training_1/RMSprop/Const_12Const*
valueB
 *    *
dtype0*
_output_shapes
: 
`
training_1/RMSprop/Const_13Const*
valueB
 *  *
dtype0*
_output_shapes
: 

*training_1/RMSprop/clip_by_value_3/MinimumMinimumtraining_1/RMSprop/add_6training_1/RMSprop/Const_13*
T0*
_output_shapes
:@

"training_1/RMSprop/clip_by_value_3Maximum*training_1/RMSprop/clip_by_value_3/Minimumtraining_1/RMSprop/Const_12*
T0*
_output_shapes
:@
j
training_1/RMSprop/Sqrt_3Sqrt"training_1/RMSprop/clip_by_value_3*
T0*
_output_shapes
:@
_
training_1/RMSprop/add_7/yConst*
valueB
 *wЬ+2*
dtype0*
_output_shapes
: 
{
training_1/RMSprop/add_7Addtraining_1/RMSprop/Sqrt_3training_1/RMSprop/add_7/y*
T0*
_output_shapes
:@

training_1/RMSprop/truediv_3RealDivtraining_1/RMSprop/mul_11training_1/RMSprop/add_7*
T0*
_output_shapes
:@
u
training_1/RMSprop/sub_7Subdense_5/bias/readtraining_1/RMSprop/truediv_3*
T0*
_output_shapes
:@
М
training_1/RMSprop/Assign_7Assigndense_5/biastraining_1/RMSprop/sub_7*
T0*
validate_shape(*
use_locking(*
_class
loc:@dense_5/bias*
_output_shapes
:@

training_1/RMSprop/mul_12MulRMSprop_1/rho/read"training_1/RMSprop/Variable_4/read*
T0*
_output_shapes

:@
_
training_1/RMSprop/sub_8/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
p
training_1/RMSprop/sub_8Subtraining_1/RMSprop/sub_8/xRMSprop_1/rho/read*
T0*
_output_shapes
: 

training_1/RMSprop/Square_4Square9training_1/RMSprop/gradients/dense_6/MatMul_grad/MatMul_1*
T0*
_output_shapes

:@

training_1/RMSprop/mul_13Multraining_1/RMSprop/sub_8training_1/RMSprop/Square_4*
T0*
_output_shapes

:@
~
training_1/RMSprop/add_8Addtraining_1/RMSprop/mul_12training_1/RMSprop/mul_13*
T0*
_output_shapes

:@
т
training_1/RMSprop/Assign_8Assigntraining_1/RMSprop/Variable_4training_1/RMSprop/add_8*
T0*
validate_shape(*
use_locking(*0
_class&
$"loc:@training_1/RMSprop/Variable_4*
_output_shapes

:@

training_1/RMSprop/mul_14MulRMSprop_1/lr/read9training_1/RMSprop/gradients/dense_6/MatMul_grad/MatMul_1*
T0*
_output_shapes

:@
`
training_1/RMSprop/Const_14Const*
valueB
 *    *
dtype0*
_output_shapes
: 
`
training_1/RMSprop/Const_15Const*
valueB
 *  *
dtype0*
_output_shapes
: 

*training_1/RMSprop/clip_by_value_4/MinimumMinimumtraining_1/RMSprop/add_8training_1/RMSprop/Const_15*
T0*
_output_shapes

:@

"training_1/RMSprop/clip_by_value_4Maximum*training_1/RMSprop/clip_by_value_4/Minimumtraining_1/RMSprop/Const_14*
T0*
_output_shapes

:@
n
training_1/RMSprop/Sqrt_4Sqrt"training_1/RMSprop/clip_by_value_4*
T0*
_output_shapes

:@
_
training_1/RMSprop/add_9/yConst*
valueB
 *wЬ+2*
dtype0*
_output_shapes
: 

training_1/RMSprop/add_9Addtraining_1/RMSprop/Sqrt_4training_1/RMSprop/add_9/y*
T0*
_output_shapes

:@

training_1/RMSprop/truediv_4RealDivtraining_1/RMSprop/mul_14training_1/RMSprop/add_9*
T0*
_output_shapes

:@
{
training_1/RMSprop/sub_9Subdense_6/kernel/readtraining_1/RMSprop/truediv_4*
T0*
_output_shapes

:@
Ф
training_1/RMSprop/Assign_9Assigndense_6/kerneltraining_1/RMSprop/sub_9*
T0*
validate_shape(*
use_locking(*!
_class
loc:@dense_6/kernel*
_output_shapes

:@
}
training_1/RMSprop/mul_15MulRMSprop_1/rho/read"training_1/RMSprop/Variable_5/read*
T0*
_output_shapes
:
`
training_1/RMSprop/sub_10/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
r
training_1/RMSprop/sub_10Subtraining_1/RMSprop/sub_10/xRMSprop_1/rho/read*
T0*
_output_shapes
: 

training_1/RMSprop/Square_5Square=training_1/RMSprop/gradients/dense_6/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:
}
training_1/RMSprop/mul_16Multraining_1/RMSprop/sub_10training_1/RMSprop/Square_5*
T0*
_output_shapes
:
{
training_1/RMSprop/add_10Addtraining_1/RMSprop/mul_15training_1/RMSprop/mul_16*
T0*
_output_shapes
:
р
training_1/RMSprop/Assign_10Assigntraining_1/RMSprop/Variable_5training_1/RMSprop/add_10*
T0*
validate_shape(*
use_locking(*0
_class&
$"loc:@training_1/RMSprop/Variable_5*
_output_shapes
:

training_1/RMSprop/mul_17MulRMSprop_1/lr/read=training_1/RMSprop/gradients/dense_6/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:
`
training_1/RMSprop/Const_16Const*
valueB
 *    *
dtype0*
_output_shapes
: 
`
training_1/RMSprop/Const_17Const*
valueB
 *  *
dtype0*
_output_shapes
: 

*training_1/RMSprop/clip_by_value_5/MinimumMinimumtraining_1/RMSprop/add_10training_1/RMSprop/Const_17*
T0*
_output_shapes
:

"training_1/RMSprop/clip_by_value_5Maximum*training_1/RMSprop/clip_by_value_5/Minimumtraining_1/RMSprop/Const_16*
T0*
_output_shapes
:
j
training_1/RMSprop/Sqrt_5Sqrt"training_1/RMSprop/clip_by_value_5*
T0*
_output_shapes
:
`
training_1/RMSprop/add_11/yConst*
valueB
 *wЬ+2*
dtype0*
_output_shapes
: 
}
training_1/RMSprop/add_11Addtraining_1/RMSprop/Sqrt_5training_1/RMSprop/add_11/y*
T0*
_output_shapes
:

training_1/RMSprop/truediv_5RealDivtraining_1/RMSprop/mul_17training_1/RMSprop/add_11*
T0*
_output_shapes
:
v
training_1/RMSprop/sub_11Subdense_6/bias/readtraining_1/RMSprop/truediv_5*
T0*
_output_shapes
:
О
training_1/RMSprop/Assign_11Assigndense_6/biastraining_1/RMSprop/sub_11*
T0*
validate_shape(*
use_locking(*
_class
loc:@dense_6/bias*
_output_shapes
:
и
training_1/group_depsNoOp^loss_1/mul%^metrics_1/mean_absolute_error/Mean_1^training_1/RMSprop/AssignAdd^training_1/RMSprop/Assign^training_1/RMSprop/Assign_1^training_1/RMSprop/Assign_2^training_1/RMSprop/Assign_3^training_1/RMSprop/Assign_4^training_1/RMSprop/Assign_5^training_1/RMSprop/Assign_6^training_1/RMSprop/Assign_7^training_1/RMSprop/Assign_8^training_1/RMSprop/Assign_9^training_1/RMSprop/Assign_10^training_1/RMSprop/Assign_11
H
group_deps_1NoOp^loss_1/mul%^metrics_1/mean_absolute_error/Mean_1

IsVariableInitialized_16IsVariableInitializeddense_4/kernel*
dtype0*!
_class
loc:@dense_4/kernel*
_output_shapes
: 

IsVariableInitialized_17IsVariableInitializeddense_4/bias*
dtype0*
_class
loc:@dense_4/bias*
_output_shapes
: 

IsVariableInitialized_18IsVariableInitializeddense_5/kernel*
dtype0*!
_class
loc:@dense_5/kernel*
_output_shapes
: 

IsVariableInitialized_19IsVariableInitializeddense_5/bias*
dtype0*
_class
loc:@dense_5/bias*
_output_shapes
: 

IsVariableInitialized_20IsVariableInitializeddense_6/kernel*
dtype0*!
_class
loc:@dense_6/kernel*
_output_shapes
: 

IsVariableInitialized_21IsVariableInitializeddense_6/bias*
dtype0*
_class
loc:@dense_6/bias*
_output_shapes
: 

IsVariableInitialized_22IsVariableInitializedRMSprop_1/lr*
dtype0*
_class
loc:@RMSprop_1/lr*
_output_shapes
: 

IsVariableInitialized_23IsVariableInitializedRMSprop_1/rho*
dtype0* 
_class
loc:@RMSprop_1/rho*
_output_shapes
: 

IsVariableInitialized_24IsVariableInitializedRMSprop_1/decay*
dtype0*"
_class
loc:@RMSprop_1/decay*
_output_shapes
: 

IsVariableInitialized_25IsVariableInitializedRMSprop_1/iterations*
dtype0	*'
_class
loc:@RMSprop_1/iterations*
_output_shapes
: 
Ѓ
IsVariableInitialized_26IsVariableInitializedtraining_1/RMSprop/Variable*
dtype0*.
_class$
" loc:@training_1/RMSprop/Variable*
_output_shapes
: 
Ї
IsVariableInitialized_27IsVariableInitializedtraining_1/RMSprop/Variable_1*
dtype0*0
_class&
$"loc:@training_1/RMSprop/Variable_1*
_output_shapes
: 
Ї
IsVariableInitialized_28IsVariableInitializedtraining_1/RMSprop/Variable_2*
dtype0*0
_class&
$"loc:@training_1/RMSprop/Variable_2*
_output_shapes
: 
Ї
IsVariableInitialized_29IsVariableInitializedtraining_1/RMSprop/Variable_3*
dtype0*0
_class&
$"loc:@training_1/RMSprop/Variable_3*
_output_shapes
: 
Ї
IsVariableInitialized_30IsVariableInitializedtraining_1/RMSprop/Variable_4*
dtype0*0
_class&
$"loc:@training_1/RMSprop/Variable_4*
_output_shapes
: 
Ї
IsVariableInitialized_31IsVariableInitializedtraining_1/RMSprop/Variable_5*
dtype0*0
_class&
$"loc:@training_1/RMSprop/Variable_5*
_output_shapes
: 
ф
init_1NoOp^dense_4/kernel/Assign^dense_4/bias/Assign^dense_5/kernel/Assign^dense_5/bias/Assign^dense_6/kernel/Assign^dense_6/bias/Assign^RMSprop_1/lr/Assign^RMSprop_1/rho/Assign^RMSprop_1/decay/Assign^RMSprop_1/iterations/Assign#^training_1/RMSprop/Variable/Assign%^training_1/RMSprop/Variable_1/Assign%^training_1/RMSprop/Variable_2/Assign%^training_1/RMSprop/Variable_3/Assign%^training_1/RMSprop/Variable_4/Assign%^training_1/RMSprop/Variable_5/Assign
e
dense_4/kernel_0/tagConst*!
valueB Bdense_4/kernel_0*
dtype0*
_output_shapes
: 
p
dense_4/kernel_0HistogramSummarydense_4/kernel_0/tagdense_4/kernel/read*
T0*
_output_shapes
: 
a
dense_4/bias_0/tagConst*
valueB Bdense_4/bias_0*
dtype0*
_output_shapes
: 
j
dense_4/bias_0HistogramSummarydense_4/bias_0/tagdense_4/bias/read*
T0*
_output_shapes
: 
[
dense_4_out/tagConst*
valueB Bdense_4_out*
dtype0*
_output_shapes
: 
_
dense_4_outHistogramSummarydense_4_out/tagdense_4/Relu*
T0*
_output_shapes
: 
e
dense_5/kernel_0/tagConst*!
valueB Bdense_5/kernel_0*
dtype0*
_output_shapes
: 
p
dense_5/kernel_0HistogramSummarydense_5/kernel_0/tagdense_5/kernel/read*
T0*
_output_shapes
: 
a
dense_5/bias_0/tagConst*
valueB Bdense_5/bias_0*
dtype0*
_output_shapes
: 
j
dense_5/bias_0HistogramSummarydense_5/bias_0/tagdense_5/bias/read*
T0*
_output_shapes
: 
[
dense_5_out/tagConst*
valueB Bdense_5_out*
dtype0*
_output_shapes
: 
_
dense_5_outHistogramSummarydense_5_out/tagdense_5/Relu*
T0*
_output_shapes
: 
e
dense_6/kernel_0/tagConst*!
valueB Bdense_6/kernel_0*
dtype0*
_output_shapes
: 
p
dense_6/kernel_0HistogramSummarydense_6/kernel_0/tagdense_6/kernel/read*
T0*
_output_shapes
: 
a
dense_6/bias_0/tagConst*
valueB Bdense_6/bias_0*
dtype0*
_output_shapes
: 
j
dense_6/bias_0HistogramSummarydense_6/bias_0/tagdense_6/bias/read*
T0*
_output_shapes
: 
[
dense_6_out/tagConst*
valueB Bdense_6_out*
dtype0*
_output_shapes
: 
b
dense_6_outHistogramSummarydense_6_out/tagdense_6/BiasAdd*
T0*
_output_shapes
: 
п
Merge_1/MergeSummaryMergeSummarydense_1/kernel_0dense_1/bias_0dense_1_outdense_2/kernel_0dense_2/bias_0dense_2_outdense_3/kernel_0dense_3/bias_0dense_3_outdense_4/kernel_0dense_4/bias_0dense_4_outdense_5/kernel_0dense_5/bias_0dense_5_outdense_6/kernel_0dense_6/bias_0dense_6_out*
N*
_output_shapes
: "ъЪї_u*     о	G}	&ЩђеыЃжAJшд
пП
,
Abs
x"T
y"T"
Ttype:

2	
:
Add
x"T
y"T
z"T"
Ttype:
2	
x
Assign
ref"T

value"T

output_ref"T"	
Ttype"
validate_shapebool("
use_lockingbool(
s
	AssignAdd
ref"T

value"T

output_ref"T" 
Ttype:
2	"
use_lockingbool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
~
BiasAddGrad
out_backprop"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
R
BroadcastGradientArgs
s0"T
s1"T
r0"T
r1"T"
Ttype0:
2	
8
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype
8
Const
output"dtype"
valuetensor"
dtypetype
S
DynamicStitch
indices*N
data"T*N
merged"T"
Nint(0"	
Ttype
4
Fill
dims

value"T
output"T"	
Ttype
?
FloorDiv
x"T
y"T
z"T"
Ttype:
2	
8
FloorMod
x"T
y"T
z"T"
Ttype:	
2	
V
HistogramSummary
tag
values"T
summary"
Ttype0:
2	
.
Identity

input"T
output"T"	
Ttype
N
IsVariableInitialized
ref"dtype
is_initialized
"
dtypetype
p
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
	2
;
Maximum
x"T
y"T
z"T"
Ttype:

2	

Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
8
MergeSummary
inputs*N
summary"
Nint(0
;
Minimum
x"T
y"T
z"T"
Ttype:

2	
=
Mul
x"T
y"T
z"T"
Ttype:
2	
.
Neg
x"T
y"T"
Ttype:

2	

NoOp
E
NotEqual
x"T
y"T
z
"
Ttype:
2	

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
2	"
Tidxtype0:
2	
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	
a
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:	
2	
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
D
Relu
features"T
activations"T"
Ttype:
2	
V
ReluGrad
	gradients"T
features"T
	backprops"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
-
Sqrt
x"T
y"T"
Ttype:

2
1
Square
x"T
y"T"
Ttype:

2	
:
Sub
x"T
y"T
z"T"
Ttype:
2	

Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
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
shared_namestring *1.5.02v1.5.0-0-g37aa430d84Є
p
dense_1_inputPlaceholder*
dtype0*
shape:џџџџџџџџџ*'
_output_shapes
:џџџџџџџџџ
m
dense_1/random_uniform/shapeConst*
valueB"   @   *
dtype0*
_output_shapes
:
_
dense_1/random_uniform/minConst*
valueB
 *!ьО*
dtype0*
_output_shapes
: 
_
dense_1/random_uniform/maxConst*
valueB
 *!ь>*
dtype0*
_output_shapes
: 
Ј
$dense_1/random_uniform/RandomUniformRandomUniformdense_1/random_uniform/shape*
seedБџх)*
seed2Дї*
dtype0*
T0*
_output_shapes

:@
z
dense_1/random_uniform/subSubdense_1/random_uniform/maxdense_1/random_uniform/min*
T0*
_output_shapes
: 

dense_1/random_uniform/mulMul$dense_1/random_uniform/RandomUniformdense_1/random_uniform/sub*
T0*
_output_shapes

:@
~
dense_1/random_uniformAdddense_1/random_uniform/muldense_1/random_uniform/min*
T0*
_output_shapes

:@

dense_1/kernel
VariableV2*
shape
:@*
dtype0*
	container *
shared_name *
_output_shapes

:@
М
dense_1/kernel/AssignAssigndense_1/kerneldense_1/random_uniform*
T0*
validate_shape(*
use_locking(*!
_class
loc:@dense_1/kernel*
_output_shapes

:@
{
dense_1/kernel/readIdentitydense_1/kernel*
T0*!
_class
loc:@dense_1/kernel*
_output_shapes

:@
Z
dense_1/ConstConst*
valueB@*    *
dtype0*
_output_shapes
:@
x
dense_1/bias
VariableV2*
shape:@*
dtype0*
	container *
shared_name *
_output_shapes
:@
Љ
dense_1/bias/AssignAssigndense_1/biasdense_1/Const*
T0*
validate_shape(*
use_locking(*
_class
loc:@dense_1/bias*
_output_shapes
:@
q
dense_1/bias/readIdentitydense_1/bias*
T0*
_class
loc:@dense_1/bias*
_output_shapes
:@

dense_1/MatMulMatMuldense_1_inputdense_1/kernel/read*
transpose_a( *
transpose_b( *
T0*'
_output_shapes
:џџџџџџџџџ@

dense_1/BiasAddBiasAdddense_1/MatMuldense_1/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:џџџџџџџџџ@
W
dense_1/ReluReludense_1/BiasAdd*
T0*'
_output_shapes
:џџџџџџџџџ@
m
dense_2/random_uniform/shapeConst*
valueB"@   @   *
dtype0*
_output_shapes
:
_
dense_2/random_uniform/minConst*
valueB
 *зГ]О*
dtype0*
_output_shapes
: 
_
dense_2/random_uniform/maxConst*
valueB
 *зГ]>*
dtype0*
_output_shapes
: 
Ј
$dense_2/random_uniform/RandomUniformRandomUniformdense_2/random_uniform/shape*
seedБџх)*
seed2џЙѕ*
dtype0*
T0*
_output_shapes

:@@
z
dense_2/random_uniform/subSubdense_2/random_uniform/maxdense_2/random_uniform/min*
T0*
_output_shapes
: 

dense_2/random_uniform/mulMul$dense_2/random_uniform/RandomUniformdense_2/random_uniform/sub*
T0*
_output_shapes

:@@
~
dense_2/random_uniformAdddense_2/random_uniform/muldense_2/random_uniform/min*
T0*
_output_shapes

:@@

dense_2/kernel
VariableV2*
shape
:@@*
dtype0*
	container *
shared_name *
_output_shapes

:@@
М
dense_2/kernel/AssignAssigndense_2/kerneldense_2/random_uniform*
T0*
validate_shape(*
use_locking(*!
_class
loc:@dense_2/kernel*
_output_shapes

:@@
{
dense_2/kernel/readIdentitydense_2/kernel*
T0*!
_class
loc:@dense_2/kernel*
_output_shapes

:@@
Z
dense_2/ConstConst*
valueB@*    *
dtype0*
_output_shapes
:@
x
dense_2/bias
VariableV2*
shape:@*
dtype0*
	container *
shared_name *
_output_shapes
:@
Љ
dense_2/bias/AssignAssigndense_2/biasdense_2/Const*
T0*
validate_shape(*
use_locking(*
_class
loc:@dense_2/bias*
_output_shapes
:@
q
dense_2/bias/readIdentitydense_2/bias*
T0*
_class
loc:@dense_2/bias*
_output_shapes
:@

dense_2/MatMulMatMuldense_1/Reludense_2/kernel/read*
transpose_a( *
transpose_b( *
T0*'
_output_shapes
:џџџџџџџџџ@

dense_2/BiasAddBiasAdddense_2/MatMuldense_2/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:џџџџџџџџџ@
W
dense_2/ReluReludense_2/BiasAdd*
T0*'
_output_shapes
:џџџџџџџџџ@
m
dense_3/random_uniform/shapeConst*
valueB"@      *
dtype0*
_output_shapes
:
_
dense_3/random_uniform/minConst*
valueB
 *О*
dtype0*
_output_shapes
: 
_
dense_3/random_uniform/maxConst*
valueB
 *>*
dtype0*
_output_shapes
: 
Ј
$dense_3/random_uniform/RandomUniformRandomUniformdense_3/random_uniform/shape*
seedБџх)*
seed2ші*
dtype0*
T0*
_output_shapes

:@
z
dense_3/random_uniform/subSubdense_3/random_uniform/maxdense_3/random_uniform/min*
T0*
_output_shapes
: 

dense_3/random_uniform/mulMul$dense_3/random_uniform/RandomUniformdense_3/random_uniform/sub*
T0*
_output_shapes

:@
~
dense_3/random_uniformAdddense_3/random_uniform/muldense_3/random_uniform/min*
T0*
_output_shapes

:@

dense_3/kernel
VariableV2*
shape
:@*
dtype0*
	container *
shared_name *
_output_shapes

:@
М
dense_3/kernel/AssignAssigndense_3/kerneldense_3/random_uniform*
T0*
validate_shape(*
use_locking(*!
_class
loc:@dense_3/kernel*
_output_shapes

:@
{
dense_3/kernel/readIdentitydense_3/kernel*
T0*!
_class
loc:@dense_3/kernel*
_output_shapes

:@
Z
dense_3/ConstConst*
valueB*    *
dtype0*
_output_shapes
:
x
dense_3/bias
VariableV2*
shape:*
dtype0*
	container *
shared_name *
_output_shapes
:
Љ
dense_3/bias/AssignAssigndense_3/biasdense_3/Const*
T0*
validate_shape(*
use_locking(*
_class
loc:@dense_3/bias*
_output_shapes
:
q
dense_3/bias/readIdentitydense_3/bias*
T0*
_class
loc:@dense_3/bias*
_output_shapes
:

dense_3/MatMulMatMuldense_2/Reludense_3/kernel/read*
transpose_a( *
transpose_b( *
T0*'
_output_shapes
:џџџџџџџџџ

dense_3/BiasAddBiasAdddense_3/MatMuldense_3/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:џџџџџџџџџ
]
RMSprop/lr/initial_valueConst*
valueB
 *o:*
dtype0*
_output_shapes
: 
n

RMSprop/lr
VariableV2*
shape: *
dtype0*
	container *
shared_name *
_output_shapes
: 
Њ
RMSprop/lr/AssignAssign
RMSprop/lrRMSprop/lr/initial_value*
T0*
validate_shape(*
use_locking(*
_class
loc:@RMSprop/lr*
_output_shapes
: 
g
RMSprop/lr/readIdentity
RMSprop/lr*
T0*
_class
loc:@RMSprop/lr*
_output_shapes
: 
^
RMSprop/rho/initial_valueConst*
valueB
 *fff?*
dtype0*
_output_shapes
: 
o
RMSprop/rho
VariableV2*
shape: *
dtype0*
	container *
shared_name *
_output_shapes
: 
Ў
RMSprop/rho/AssignAssignRMSprop/rhoRMSprop/rho/initial_value*
T0*
validate_shape(*
use_locking(*
_class
loc:@RMSprop/rho*
_output_shapes
: 
j
RMSprop/rho/readIdentityRMSprop/rho*
T0*
_class
loc:@RMSprop/rho*
_output_shapes
: 
`
RMSprop/decay/initial_valueConst*
valueB
 *    *
dtype0*
_output_shapes
: 
q
RMSprop/decay
VariableV2*
shape: *
dtype0*
	container *
shared_name *
_output_shapes
: 
Ж
RMSprop/decay/AssignAssignRMSprop/decayRMSprop/decay/initial_value*
T0*
validate_shape(*
use_locking(* 
_class
loc:@RMSprop/decay*
_output_shapes
: 
p
RMSprop/decay/readIdentityRMSprop/decay*
T0* 
_class
loc:@RMSprop/decay*
_output_shapes
: 
b
 RMSprop/iterations/initial_valueConst*
value	B	 R *
dtype0	*
_output_shapes
: 
v
RMSprop/iterations
VariableV2*
shape: *
dtype0	*
	container *
shared_name *
_output_shapes
: 
Ъ
RMSprop/iterations/AssignAssignRMSprop/iterations RMSprop/iterations/initial_value*
T0	*
validate_shape(*
use_locking(*%
_class
loc:@RMSprop/iterations*
_output_shapes
: 

RMSprop/iterations/readIdentityRMSprop/iterations*
T0	*%
_class
loc:@RMSprop/iterations*
_output_shapes
: 

dense_3_targetPlaceholder*
dtype0*%
shape:џџџџџџџџџџџџџџџџџџ*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
q
dense_3_sample_weightsPlaceholder*
dtype0*
shape:џџџџџџџџџ*#
_output_shapes
:џџџџџџџџџ
x
loss/dense_3_loss/subSubdense_3/BiasAdddense_3_target*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
t
loss/dense_3_loss/SquareSquareloss/dense_3_loss/sub*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
s
(loss/dense_3_loss/Mean/reduction_indicesConst*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 
­
loss/dense_3_loss/MeanMeanloss/dense_3_loss/Square(loss/dense_3_loss/Mean/reduction_indices*
	keep_dims( *
T0*

Tidx0*#
_output_shapes
:џџџџџџџџџ
m
*loss/dense_3_loss/Mean_1/reduction_indicesConst*
valueB *
dtype0*
_output_shapes
: 
Џ
loss/dense_3_loss/Mean_1Meanloss/dense_3_loss/Mean*loss/dense_3_loss/Mean_1/reduction_indices*
	keep_dims( *
T0*

Tidx0*#
_output_shapes
:џџџџџџџџџ
|
loss/dense_3_loss/mulMulloss/dense_3_loss/Mean_1dense_3_sample_weights*
T0*#
_output_shapes
:џџџџџџџџџ
a
loss/dense_3_loss/NotEqual/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 

loss/dense_3_loss/NotEqualNotEqualdense_3_sample_weightsloss/dense_3_loss/NotEqual/y*
T0*#
_output_shapes
:џџџџџџџџџ
w
loss/dense_3_loss/CastCastloss/dense_3_loss/NotEqual*

SrcT0
*

DstT0*#
_output_shapes
:џџџџџџџџџ
a
loss/dense_3_loss/ConstConst*
valueB: *
dtype0*
_output_shapes
:

loss/dense_3_loss/Mean_2Meanloss/dense_3_loss/Castloss/dense_3_loss/Const*
	keep_dims( *
T0*

Tidx0*
_output_shapes
: 

loss/dense_3_loss/truedivRealDivloss/dense_3_loss/mulloss/dense_3_loss/Mean_2*
T0*#
_output_shapes
:џџџџџџџџџ
c
loss/dense_3_loss/Const_1Const*
valueB: *
dtype0*
_output_shapes
:

loss/dense_3_loss/Mean_3Meanloss/dense_3_loss/truedivloss/dense_3_loss/Const_1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
: 
O

loss/mul/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
V
loss/mulMul
loss/mul/xloss/dense_3_loss/Mean_3*
T0*
_output_shapes
: 

metrics/mean_absolute_error/subSubdense_3/BiasAdddense_3_target*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ

metrics/mean_absolute_error/AbsAbsmetrics/mean_absolute_error/sub*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
}
2metrics/mean_absolute_error/Mean/reduction_indicesConst*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 
Ш
 metrics/mean_absolute_error/MeanMeanmetrics/mean_absolute_error/Abs2metrics/mean_absolute_error/Mean/reduction_indices*
	keep_dims( *
T0*

Tidx0*#
_output_shapes
:џџџџџџџџџ
k
!metrics/mean_absolute_error/ConstConst*
valueB: *
dtype0*
_output_shapes
:
­
"metrics/mean_absolute_error/Mean_1Mean metrics/mean_absolute_error/Mean!metrics/mean_absolute_error/Const*
	keep_dims( *
T0*

Tidx0*
_output_shapes
: 

 training/RMSprop/gradients/ShapeConst*
valueB *
dtype0*
_class
loc:@loss/mul*
_output_shapes
: 

$training/RMSprop/gradients/grad_ys_0Const*
valueB
 *  ?*
dtype0*
_class
loc:@loss/mul*
_output_shapes
: 
­
training/RMSprop/gradients/FillFill training/RMSprop/gradients/Shape$training/RMSprop/gradients/grad_ys_0*
T0*
_class
loc:@loss/mul*
_output_shapes
: 
Ќ
,training/RMSprop/gradients/loss/mul_grad/MulMultraining/RMSprop/gradients/Fillloss/dense_3_loss/Mean_3*
T0*
_class
loc:@loss/mul*
_output_shapes
: 
 
.training/RMSprop/gradients/loss/mul_grad/Mul_1Multraining/RMSprop/gradients/Fill
loss/mul/x*
T0*
_class
loc:@loss/mul*
_output_shapes
: 
Н
Ftraining/RMSprop/gradients/loss/dense_3_loss/Mean_3_grad/Reshape/shapeConst*
valueB:*
dtype0*+
_class!
loc:@loss/dense_3_loss/Mean_3*
_output_shapes
:
Ѓ
@training/RMSprop/gradients/loss/dense_3_loss/Mean_3_grad/ReshapeReshape.training/RMSprop/gradients/loss/mul_grad/Mul_1Ftraining/RMSprop/gradients/loss/dense_3_loss/Mean_3_grad/Reshape/shape*
T0*
Tshape0*+
_class!
loc:@loss/dense_3_loss/Mean_3*
_output_shapes
:
Ф
>training/RMSprop/gradients/loss/dense_3_loss/Mean_3_grad/ShapeShapeloss/dense_3_loss/truediv*
T0*
out_type0*+
_class!
loc:@loss/dense_3_loss/Mean_3*
_output_shapes
:
Д
=training/RMSprop/gradients/loss/dense_3_loss/Mean_3_grad/TileTile@training/RMSprop/gradients/loss/dense_3_loss/Mean_3_grad/Reshape>training/RMSprop/gradients/loss/dense_3_loss/Mean_3_grad/Shape*
T0*

Tmultiples0*+
_class!
loc:@loss/dense_3_loss/Mean_3*#
_output_shapes
:џџџџџџџџџ
Ц
@training/RMSprop/gradients/loss/dense_3_loss/Mean_3_grad/Shape_1Shapeloss/dense_3_loss/truediv*
T0*
out_type0*+
_class!
loc:@loss/dense_3_loss/Mean_3*
_output_shapes
:
А
@training/RMSprop/gradients/loss/dense_3_loss/Mean_3_grad/Shape_2Const*
valueB *
dtype0*+
_class!
loc:@loss/dense_3_loss/Mean_3*
_output_shapes
: 
Е
>training/RMSprop/gradients/loss/dense_3_loss/Mean_3_grad/ConstConst*
valueB: *
dtype0*+
_class!
loc:@loss/dense_3_loss/Mean_3*
_output_shapes
:
В
=training/RMSprop/gradients/loss/dense_3_loss/Mean_3_grad/ProdProd@training/RMSprop/gradients/loss/dense_3_loss/Mean_3_grad/Shape_1>training/RMSprop/gradients/loss/dense_3_loss/Mean_3_grad/Const*
	keep_dims( *
T0*

Tidx0*+
_class!
loc:@loss/dense_3_loss/Mean_3*
_output_shapes
: 
З
@training/RMSprop/gradients/loss/dense_3_loss/Mean_3_grad/Const_1Const*
valueB: *
dtype0*+
_class!
loc:@loss/dense_3_loss/Mean_3*
_output_shapes
:
Ж
?training/RMSprop/gradients/loss/dense_3_loss/Mean_3_grad/Prod_1Prod@training/RMSprop/gradients/loss/dense_3_loss/Mean_3_grad/Shape_2@training/RMSprop/gradients/loss/dense_3_loss/Mean_3_grad/Const_1*
	keep_dims( *
T0*

Tidx0*+
_class!
loc:@loss/dense_3_loss/Mean_3*
_output_shapes
: 
Б
Btraining/RMSprop/gradients/loss/dense_3_loss/Mean_3_grad/Maximum/yConst*
value	B :*
dtype0*+
_class!
loc:@loss/dense_3_loss/Mean_3*
_output_shapes
: 

@training/RMSprop/gradients/loss/dense_3_loss/Mean_3_grad/MaximumMaximum?training/RMSprop/gradients/loss/dense_3_loss/Mean_3_grad/Prod_1Btraining/RMSprop/gradients/loss/dense_3_loss/Mean_3_grad/Maximum/y*
T0*+
_class!
loc:@loss/dense_3_loss/Mean_3*
_output_shapes
: 

Atraining/RMSprop/gradients/loss/dense_3_loss/Mean_3_grad/floordivFloorDiv=training/RMSprop/gradients/loss/dense_3_loss/Mean_3_grad/Prod@training/RMSprop/gradients/loss/dense_3_loss/Mean_3_grad/Maximum*
T0*+
_class!
loc:@loss/dense_3_loss/Mean_3*
_output_shapes
: 
х
=training/RMSprop/gradients/loss/dense_3_loss/Mean_3_grad/CastCastAtraining/RMSprop/gradients/loss/dense_3_loss/Mean_3_grad/floordiv*

SrcT0*

DstT0*+
_class!
loc:@loss/dense_3_loss/Mean_3*
_output_shapes
: 
Є
@training/RMSprop/gradients/loss/dense_3_loss/Mean_3_grad/truedivRealDiv=training/RMSprop/gradients/loss/dense_3_loss/Mean_3_grad/Tile=training/RMSprop/gradients/loss/dense_3_loss/Mean_3_grad/Cast*
T0*+
_class!
loc:@loss/dense_3_loss/Mean_3*#
_output_shapes
:џџџџџџџџџ
Т
?training/RMSprop/gradients/loss/dense_3_loss/truediv_grad/ShapeShapeloss/dense_3_loss/mul*
T0*
out_type0*,
_class"
 loc:@loss/dense_3_loss/truediv*
_output_shapes
:
В
Atraining/RMSprop/gradients/loss/dense_3_loss/truediv_grad/Shape_1Const*
valueB *
dtype0*,
_class"
 loc:@loss/dense_3_loss/truediv*
_output_shapes
: 
з
Otraining/RMSprop/gradients/loss/dense_3_loss/truediv_grad/BroadcastGradientArgsBroadcastGradientArgs?training/RMSprop/gradients/loss/dense_3_loss/truediv_grad/ShapeAtraining/RMSprop/gradients/loss/dense_3_loss/truediv_grad/Shape_1*
T0*,
_class"
 loc:@loss/dense_3_loss/truediv*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ

Atraining/RMSprop/gradients/loss/dense_3_loss/truediv_grad/RealDivRealDiv@training/RMSprop/gradients/loss/dense_3_loss/Mean_3_grad/truedivloss/dense_3_loss/Mean_2*
T0*,
_class"
 loc:@loss/dense_3_loss/truediv*#
_output_shapes
:џџџџџџџџџ
Ц
=training/RMSprop/gradients/loss/dense_3_loss/truediv_grad/SumSumAtraining/RMSprop/gradients/loss/dense_3_loss/truediv_grad/RealDivOtraining/RMSprop/gradients/loss/dense_3_loss/truediv_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*,
_class"
 loc:@loss/dense_3_loss/truediv*
_output_shapes
:
Ж
Atraining/RMSprop/gradients/loss/dense_3_loss/truediv_grad/ReshapeReshape=training/RMSprop/gradients/loss/dense_3_loss/truediv_grad/Sum?training/RMSprop/gradients/loss/dense_3_loss/truediv_grad/Shape*
T0*
Tshape0*,
_class"
 loc:@loss/dense_3_loss/truediv*#
_output_shapes
:џџџџџџџџџ
З
=training/RMSprop/gradients/loss/dense_3_loss/truediv_grad/NegNegloss/dense_3_loss/mul*
T0*,
_class"
 loc:@loss/dense_3_loss/truediv*#
_output_shapes
:џџџџџџџџџ

Ctraining/RMSprop/gradients/loss/dense_3_loss/truediv_grad/RealDiv_1RealDiv=training/RMSprop/gradients/loss/dense_3_loss/truediv_grad/Negloss/dense_3_loss/Mean_2*
T0*,
_class"
 loc:@loss/dense_3_loss/truediv*#
_output_shapes
:џџџџџџџџџ

Ctraining/RMSprop/gradients/loss/dense_3_loss/truediv_grad/RealDiv_2RealDivCtraining/RMSprop/gradients/loss/dense_3_loss/truediv_grad/RealDiv_1loss/dense_3_loss/Mean_2*
T0*,
_class"
 loc:@loss/dense_3_loss/truediv*#
_output_shapes
:џџџџџџџџџ
Ї
=training/RMSprop/gradients/loss/dense_3_loss/truediv_grad/mulMul@training/RMSprop/gradients/loss/dense_3_loss/Mean_3_grad/truedivCtraining/RMSprop/gradients/loss/dense_3_loss/truediv_grad/RealDiv_2*
T0*,
_class"
 loc:@loss/dense_3_loss/truediv*#
_output_shapes
:џџџџџџџџџ
Ц
?training/RMSprop/gradients/loss/dense_3_loss/truediv_grad/Sum_1Sum=training/RMSprop/gradients/loss/dense_3_loss/truediv_grad/mulQtraining/RMSprop/gradients/loss/dense_3_loss/truediv_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*,
_class"
 loc:@loss/dense_3_loss/truediv*
_output_shapes
:
Џ
Ctraining/RMSprop/gradients/loss/dense_3_loss/truediv_grad/Reshape_1Reshape?training/RMSprop/gradients/loss/dense_3_loss/truediv_grad/Sum_1Atraining/RMSprop/gradients/loss/dense_3_loss/truediv_grad/Shape_1*
T0*
Tshape0*,
_class"
 loc:@loss/dense_3_loss/truediv*
_output_shapes
: 
Н
;training/RMSprop/gradients/loss/dense_3_loss/mul_grad/ShapeShapeloss/dense_3_loss/Mean_1*
T0*
out_type0*(
_class
loc:@loss/dense_3_loss/mul*
_output_shapes
:
Н
=training/RMSprop/gradients/loss/dense_3_loss/mul_grad/Shape_1Shapedense_3_sample_weights*
T0*
out_type0*(
_class
loc:@loss/dense_3_loss/mul*
_output_shapes
:
Ч
Ktraining/RMSprop/gradients/loss/dense_3_loss/mul_grad/BroadcastGradientArgsBroadcastGradientArgs;training/RMSprop/gradients/loss/dense_3_loss/mul_grad/Shape=training/RMSprop/gradients/loss/dense_3_loss/mul_grad/Shape_1*
T0*(
_class
loc:@loss/dense_3_loss/mul*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
ѓ
9training/RMSprop/gradients/loss/dense_3_loss/mul_grad/mulMulAtraining/RMSprop/gradients/loss/dense_3_loss/truediv_grad/Reshapedense_3_sample_weights*
T0*(
_class
loc:@loss/dense_3_loss/mul*#
_output_shapes
:џџџџџџџџџ
В
9training/RMSprop/gradients/loss/dense_3_loss/mul_grad/SumSum9training/RMSprop/gradients/loss/dense_3_loss/mul_grad/mulKtraining/RMSprop/gradients/loss/dense_3_loss/mul_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*(
_class
loc:@loss/dense_3_loss/mul*
_output_shapes
:
І
=training/RMSprop/gradients/loss/dense_3_loss/mul_grad/ReshapeReshape9training/RMSprop/gradients/loss/dense_3_loss/mul_grad/Sum;training/RMSprop/gradients/loss/dense_3_loss/mul_grad/Shape*
T0*
Tshape0*(
_class
loc:@loss/dense_3_loss/mul*#
_output_shapes
:џџџџџџџџџ
ї
;training/RMSprop/gradients/loss/dense_3_loss/mul_grad/mul_1Mulloss/dense_3_loss/Mean_1Atraining/RMSprop/gradients/loss/dense_3_loss/truediv_grad/Reshape*
T0*(
_class
loc:@loss/dense_3_loss/mul*#
_output_shapes
:џџџџџџџџџ
И
;training/RMSprop/gradients/loss/dense_3_loss/mul_grad/Sum_1Sum;training/RMSprop/gradients/loss/dense_3_loss/mul_grad/mul_1Mtraining/RMSprop/gradients/loss/dense_3_loss/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*(
_class
loc:@loss/dense_3_loss/mul*
_output_shapes
:
Ќ
?training/RMSprop/gradients/loss/dense_3_loss/mul_grad/Reshape_1Reshape;training/RMSprop/gradients/loss/dense_3_loss/mul_grad/Sum_1=training/RMSprop/gradients/loss/dense_3_loss/mul_grad/Shape_1*
T0*
Tshape0*(
_class
loc:@loss/dense_3_loss/mul*#
_output_shapes
:џџџџџџџџџ
С
>training/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/ShapeShapeloss/dense_3_loss/Mean*
T0*
out_type0*+
_class!
loc:@loss/dense_3_loss/Mean_1*
_output_shapes
:
Ќ
=training/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/SizeConst*
value	B :*
dtype0*+
_class!
loc:@loss/dense_3_loss/Mean_1*
_output_shapes
: 
ў
<training/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/addAdd*loss/dense_3_loss/Mean_1/reduction_indices=training/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/Size*
T0*+
_class!
loc:@loss/dense_3_loss/Mean_1*
_output_shapes
: 

<training/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/modFloorMod<training/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/add=training/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/Size*
T0*+
_class!
loc:@loss/dense_3_loss/Mean_1*
_output_shapes
: 
З
@training/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/Shape_1Const*
valueB: *
dtype0*+
_class!
loc:@loss/dense_3_loss/Mean_1*
_output_shapes
:
Г
Dtraining/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/range/startConst*
value	B : *
dtype0*+
_class!
loc:@loss/dense_3_loss/Mean_1*
_output_shapes
: 
Г
Dtraining/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/range/deltaConst*
value	B :*
dtype0*+
_class!
loc:@loss/dense_3_loss/Mean_1*
_output_shapes
: 
ч
>training/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/rangeRangeDtraining/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/range/start=training/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/SizeDtraining/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/range/delta*

Tidx0*+
_class!
loc:@loss/dense_3_loss/Mean_1*
_output_shapes
:
В
Ctraining/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/Fill/valueConst*
value	B :*
dtype0*+
_class!
loc:@loss/dense_3_loss/Mean_1*
_output_shapes
: 

=training/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/FillFill@training/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/Shape_1Ctraining/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/Fill/value*
T0*+
_class!
loc:@loss/dense_3_loss/Mean_1*
_output_shapes
: 
И
Ftraining/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/DynamicStitchDynamicStitch>training/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/range<training/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/mod>training/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/Shape=training/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/Fill*
N*
T0*+
_class!
loc:@loss/dense_3_loss/Mean_1*#
_output_shapes
:џџџџџџџџџ
Б
Btraining/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/Maximum/yConst*
value	B :*
dtype0*+
_class!
loc:@loss/dense_3_loss/Mean_1*
_output_shapes
: 
В
@training/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/MaximumMaximumFtraining/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/DynamicStitchBtraining/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/Maximum/y*
T0*+
_class!
loc:@loss/dense_3_loss/Mean_1*#
_output_shapes
:џџџџџџџџџ
Њ
Atraining/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/floordivFloorDiv>training/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/Shape@training/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/Maximum*
T0*+
_class!
loc:@loss/dense_3_loss/Mean_1*#
_output_shapes
:џџџџџџџџџ
А
@training/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/ReshapeReshape=training/RMSprop/gradients/loss/dense_3_loss/mul_grad/ReshapeFtraining/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/DynamicStitch*
T0*
Tshape0*+
_class!
loc:@loss/dense_3_loss/Mean_1*
_output_shapes
:
Ќ
=training/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/TileTile@training/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/ReshapeAtraining/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/floordiv*
T0*

Tmultiples0*+
_class!
loc:@loss/dense_3_loss/Mean_1*
_output_shapes
:
У
@training/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/Shape_2Shapeloss/dense_3_loss/Mean*
T0*
out_type0*+
_class!
loc:@loss/dense_3_loss/Mean_1*
_output_shapes
:
Х
@training/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/Shape_3Shapeloss/dense_3_loss/Mean_1*
T0*
out_type0*+
_class!
loc:@loss/dense_3_loss/Mean_1*
_output_shapes
:
Е
>training/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/ConstConst*
valueB: *
dtype0*+
_class!
loc:@loss/dense_3_loss/Mean_1*
_output_shapes
:
В
=training/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/ProdProd@training/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/Shape_2>training/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/Const*
	keep_dims( *
T0*

Tidx0*+
_class!
loc:@loss/dense_3_loss/Mean_1*
_output_shapes
: 
З
@training/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/Const_1Const*
valueB: *
dtype0*+
_class!
loc:@loss/dense_3_loss/Mean_1*
_output_shapes
:
Ж
?training/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/Prod_1Prod@training/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/Shape_3@training/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/Const_1*
	keep_dims( *
T0*

Tidx0*+
_class!
loc:@loss/dense_3_loss/Mean_1*
_output_shapes
: 
Г
Dtraining/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/Maximum_1/yConst*
value	B :*
dtype0*+
_class!
loc:@loss/dense_3_loss/Mean_1*
_output_shapes
: 
Ђ
Btraining/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/Maximum_1Maximum?training/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/Prod_1Dtraining/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/Maximum_1/y*
T0*+
_class!
loc:@loss/dense_3_loss/Mean_1*
_output_shapes
: 
 
Ctraining/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/floordiv_1FloorDiv=training/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/ProdBtraining/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/Maximum_1*
T0*+
_class!
loc:@loss/dense_3_loss/Mean_1*
_output_shapes
: 
ч
=training/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/CastCastCtraining/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/floordiv_1*

SrcT0*

DstT0*+
_class!
loc:@loss/dense_3_loss/Mean_1*
_output_shapes
: 
Є
@training/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/truedivRealDiv=training/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/Tile=training/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/Cast*
T0*+
_class!
loc:@loss/dense_3_loss/Mean_1*#
_output_shapes
:џџџџџџџџџ
П
<training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/ShapeShapeloss/dense_3_loss/Square*
T0*
out_type0*)
_class
loc:@loss/dense_3_loss/Mean*
_output_shapes
:
Ј
;training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/SizeConst*
value	B :*
dtype0*)
_class
loc:@loss/dense_3_loss/Mean*
_output_shapes
: 
є
:training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/addAdd(loss/dense_3_loss/Mean/reduction_indices;training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/Size*
T0*)
_class
loc:@loss/dense_3_loss/Mean*
_output_shapes
: 

:training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/modFloorMod:training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/add;training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/Size*
T0*)
_class
loc:@loss/dense_3_loss/Mean*
_output_shapes
: 
Ќ
>training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/Shape_1Const*
valueB *
dtype0*)
_class
loc:@loss/dense_3_loss/Mean*
_output_shapes
: 
Џ
Btraining/RMSprop/gradients/loss/dense_3_loss/Mean_grad/range/startConst*
value	B : *
dtype0*)
_class
loc:@loss/dense_3_loss/Mean*
_output_shapes
: 
Џ
Btraining/RMSprop/gradients/loss/dense_3_loss/Mean_grad/range/deltaConst*
value	B :*
dtype0*)
_class
loc:@loss/dense_3_loss/Mean*
_output_shapes
: 
н
<training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/rangeRangeBtraining/RMSprop/gradients/loss/dense_3_loss/Mean_grad/range/start;training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/SizeBtraining/RMSprop/gradients/loss/dense_3_loss/Mean_grad/range/delta*

Tidx0*)
_class
loc:@loss/dense_3_loss/Mean*
_output_shapes
:
Ў
Atraining/RMSprop/gradients/loss/dense_3_loss/Mean_grad/Fill/valueConst*
value	B :*
dtype0*)
_class
loc:@loss/dense_3_loss/Mean*
_output_shapes
: 

;training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/FillFill>training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/Shape_1Atraining/RMSprop/gradients/loss/dense_3_loss/Mean_grad/Fill/value*
T0*)
_class
loc:@loss/dense_3_loss/Mean*
_output_shapes
: 
Ќ
Dtraining/RMSprop/gradients/loss/dense_3_loss/Mean_grad/DynamicStitchDynamicStitch<training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/range:training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/mod<training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/Shape;training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/Fill*
N*
T0*)
_class
loc:@loss/dense_3_loss/Mean*#
_output_shapes
:џџџџџџџџџ
­
@training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/Maximum/yConst*
value	B :*
dtype0*)
_class
loc:@loss/dense_3_loss/Mean*
_output_shapes
: 
Њ
>training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/MaximumMaximumDtraining/RMSprop/gradients/loss/dense_3_loss/Mean_grad/DynamicStitch@training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/Maximum/y*
T0*)
_class
loc:@loss/dense_3_loss/Mean*#
_output_shapes
:џџџџџџџџџ

?training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/floordivFloorDiv<training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/Shape>training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/Maximum*
T0*)
_class
loc:@loss/dense_3_loss/Mean*
_output_shapes
:
­
>training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/ReshapeReshape@training/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/truedivDtraining/RMSprop/gradients/loss/dense_3_loss/Mean_grad/DynamicStitch*
T0*
Tshape0*)
_class
loc:@loss/dense_3_loss/Mean*
_output_shapes
:
М
;training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/TileTile>training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/Reshape?training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/floordiv*
T0*

Tmultiples0*)
_class
loc:@loss/dense_3_loss/Mean*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
С
>training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/Shape_2Shapeloss/dense_3_loss/Square*
T0*
out_type0*)
_class
loc:@loss/dense_3_loss/Mean*
_output_shapes
:
П
>training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/Shape_3Shapeloss/dense_3_loss/Mean*
T0*
out_type0*)
_class
loc:@loss/dense_3_loss/Mean*
_output_shapes
:
Б
<training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/ConstConst*
valueB: *
dtype0*)
_class
loc:@loss/dense_3_loss/Mean*
_output_shapes
:
Њ
;training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/ProdProd>training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/Shape_2<training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/Const*
	keep_dims( *
T0*

Tidx0*)
_class
loc:@loss/dense_3_loss/Mean*
_output_shapes
: 
Г
>training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/Const_1Const*
valueB: *
dtype0*)
_class
loc:@loss/dense_3_loss/Mean*
_output_shapes
:
Ў
=training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/Prod_1Prod>training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/Shape_3>training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/Const_1*
	keep_dims( *
T0*

Tidx0*)
_class
loc:@loss/dense_3_loss/Mean*
_output_shapes
: 
Џ
Btraining/RMSprop/gradients/loss/dense_3_loss/Mean_grad/Maximum_1/yConst*
value	B :*
dtype0*)
_class
loc:@loss/dense_3_loss/Mean*
_output_shapes
: 

@training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/Maximum_1Maximum=training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/Prod_1Btraining/RMSprop/gradients/loss/dense_3_loss/Mean_grad/Maximum_1/y*
T0*)
_class
loc:@loss/dense_3_loss/Mean*
_output_shapes
: 

Atraining/RMSprop/gradients/loss/dense_3_loss/Mean_grad/floordiv_1FloorDiv;training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/Prod@training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/Maximum_1*
T0*)
_class
loc:@loss/dense_3_loss/Mean*
_output_shapes
: 
с
;training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/CastCastAtraining/RMSprop/gradients/loss/dense_3_loss/Mean_grad/floordiv_1*

SrcT0*

DstT0*)
_class
loc:@loss/dense_3_loss/Mean*
_output_shapes
: 
Љ
>training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/truedivRealDiv;training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/Tile;training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/Cast*
T0*)
_class
loc:@loss/dense_3_loss/Mean*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
ё
>training/RMSprop/gradients/loss/dense_3_loss/Square_grad/mul/xConst?^training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/truediv*
valueB
 *   @*
dtype0*+
_class!
loc:@loss/dense_3_loss/Square*
_output_shapes
: 

<training/RMSprop/gradients/loss/dense_3_loss/Square_grad/mulMul>training/RMSprop/gradients/loss/dense_3_loss/Square_grad/mul/xloss/dense_3_loss/sub*
T0*+
_class!
loc:@loss/dense_3_loss/Square*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
Ћ
>training/RMSprop/gradients/loss/dense_3_loss/Square_grad/mul_1Mul>training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/truediv<training/RMSprop/gradients/loss/dense_3_loss/Square_grad/mul*
T0*+
_class!
loc:@loss/dense_3_loss/Square*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
Д
;training/RMSprop/gradients/loss/dense_3_loss/sub_grad/ShapeShapedense_3/BiasAdd*
T0*
out_type0*(
_class
loc:@loss/dense_3_loss/sub*
_output_shapes
:
Е
=training/RMSprop/gradients/loss/dense_3_loss/sub_grad/Shape_1Shapedense_3_target*
T0*
out_type0*(
_class
loc:@loss/dense_3_loss/sub*
_output_shapes
:
Ч
Ktraining/RMSprop/gradients/loss/dense_3_loss/sub_grad/BroadcastGradientArgsBroadcastGradientArgs;training/RMSprop/gradients/loss/dense_3_loss/sub_grad/Shape=training/RMSprop/gradients/loss/dense_3_loss/sub_grad/Shape_1*
T0*(
_class
loc:@loss/dense_3_loss/sub*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
З
9training/RMSprop/gradients/loss/dense_3_loss/sub_grad/SumSum>training/RMSprop/gradients/loss/dense_3_loss/Square_grad/mul_1Ktraining/RMSprop/gradients/loss/dense_3_loss/sub_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*(
_class
loc:@loss/dense_3_loss/sub*
_output_shapes
:
Њ
=training/RMSprop/gradients/loss/dense_3_loss/sub_grad/ReshapeReshape9training/RMSprop/gradients/loss/dense_3_loss/sub_grad/Sum;training/RMSprop/gradients/loss/dense_3_loss/sub_grad/Shape*
T0*
Tshape0*(
_class
loc:@loss/dense_3_loss/sub*'
_output_shapes
:џџџџџџџџџ
Л
;training/RMSprop/gradients/loss/dense_3_loss/sub_grad/Sum_1Sum>training/RMSprop/gradients/loss/dense_3_loss/Square_grad/mul_1Mtraining/RMSprop/gradients/loss/dense_3_loss/sub_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*(
_class
loc:@loss/dense_3_loss/sub*
_output_shapes
:
Ъ
9training/RMSprop/gradients/loss/dense_3_loss/sub_grad/NegNeg;training/RMSprop/gradients/loss/dense_3_loss/sub_grad/Sum_1*
T0*(
_class
loc:@loss/dense_3_loss/sub*
_output_shapes
:
З
?training/RMSprop/gradients/loss/dense_3_loss/sub_grad/Reshape_1Reshape9training/RMSprop/gradients/loss/dense_3_loss/sub_grad/Neg=training/RMSprop/gradients/loss/dense_3_loss/sub_grad/Shape_1*
T0*
Tshape0*(
_class
loc:@loss/dense_3_loss/sub*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
щ
;training/RMSprop/gradients/dense_3/BiasAdd_grad/BiasAddGradBiasAddGrad=training/RMSprop/gradients/loss/dense_3_loss/sub_grad/Reshape*
T0*
data_formatNHWC*"
_class
loc:@dense_3/BiasAdd*
_output_shapes
:

5training/RMSprop/gradients/dense_3/MatMul_grad/MatMulMatMul=training/RMSprop/gradients/loss/dense_3_loss/sub_grad/Reshapedense_3/kernel/read*
transpose_a( *
transpose_b(*
T0*!
_class
loc:@dense_3/MatMul*'
_output_shapes
:џџџџџџџџџ@

7training/RMSprop/gradients/dense_3/MatMul_grad/MatMul_1MatMuldense_2/Relu=training/RMSprop/gradients/loss/dense_3_loss/sub_grad/Reshape*
transpose_a(*
transpose_b( *
T0*!
_class
loc:@dense_3/MatMul*
_output_shapes

:@
й
5training/RMSprop/gradients/dense_2/Relu_grad/ReluGradReluGrad5training/RMSprop/gradients/dense_3/MatMul_grad/MatMuldense_2/Relu*
T0*
_class
loc:@dense_2/Relu*'
_output_shapes
:џџџџџџџџџ@
с
;training/RMSprop/gradients/dense_2/BiasAdd_grad/BiasAddGradBiasAddGrad5training/RMSprop/gradients/dense_2/Relu_grad/ReluGrad*
T0*
data_formatNHWC*"
_class
loc:@dense_2/BiasAdd*
_output_shapes
:@

5training/RMSprop/gradients/dense_2/MatMul_grad/MatMulMatMul5training/RMSprop/gradients/dense_2/Relu_grad/ReluGraddense_2/kernel/read*
transpose_a( *
transpose_b(*
T0*!
_class
loc:@dense_2/MatMul*'
_output_shapes
:џџџџџџџџџ@
ј
7training/RMSprop/gradients/dense_2/MatMul_grad/MatMul_1MatMuldense_1/Relu5training/RMSprop/gradients/dense_2/Relu_grad/ReluGrad*
transpose_a(*
transpose_b( *
T0*!
_class
loc:@dense_2/MatMul*
_output_shapes

:@@
й
5training/RMSprop/gradients/dense_1/Relu_grad/ReluGradReluGrad5training/RMSprop/gradients/dense_2/MatMul_grad/MatMuldense_1/Relu*
T0*
_class
loc:@dense_1/Relu*'
_output_shapes
:џџџџџџџџџ@
с
;training/RMSprop/gradients/dense_1/BiasAdd_grad/BiasAddGradBiasAddGrad5training/RMSprop/gradients/dense_1/Relu_grad/ReluGrad*
T0*
data_formatNHWC*"
_class
loc:@dense_1/BiasAdd*
_output_shapes
:@

5training/RMSprop/gradients/dense_1/MatMul_grad/MatMulMatMul5training/RMSprop/gradients/dense_1/Relu_grad/ReluGraddense_1/kernel/read*
transpose_a( *
transpose_b(*
T0*!
_class
loc:@dense_1/MatMul*'
_output_shapes
:џџџџџџџџџ
љ
7training/RMSprop/gradients/dense_1/MatMul_grad/MatMul_1MatMuldense_1_input5training/RMSprop/gradients/dense_1/Relu_grad/ReluGrad*
transpose_a(*
transpose_b( *
T0*!
_class
loc:@dense_1/MatMul*
_output_shapes

:@
k
training/RMSprop/ConstConst*
valueB@*    *
dtype0*
_output_shapes

:@

training/RMSprop/Variable
VariableV2*
shape
:@*
dtype0*
	container *
shared_name *
_output_shapes

:@
н
 training/RMSprop/Variable/AssignAssigntraining/RMSprop/Variabletraining/RMSprop/Const*
T0*
validate_shape(*
use_locking(*,
_class"
 loc:@training/RMSprop/Variable*
_output_shapes

:@

training/RMSprop/Variable/readIdentitytraining/RMSprop/Variable*
T0*,
_class"
 loc:@training/RMSprop/Variable*
_output_shapes

:@
e
training/RMSprop/Const_1Const*
valueB@*    *
dtype0*
_output_shapes
:@

training/RMSprop/Variable_1
VariableV2*
shape:@*
dtype0*
	container *
shared_name *
_output_shapes
:@
с
"training/RMSprop/Variable_1/AssignAssigntraining/RMSprop/Variable_1training/RMSprop/Const_1*
T0*
validate_shape(*
use_locking(*.
_class$
" loc:@training/RMSprop/Variable_1*
_output_shapes
:@

 training/RMSprop/Variable_1/readIdentitytraining/RMSprop/Variable_1*
T0*.
_class$
" loc:@training/RMSprop/Variable_1*
_output_shapes
:@
m
training/RMSprop/Const_2Const*
valueB@@*    *
dtype0*
_output_shapes

:@@

training/RMSprop/Variable_2
VariableV2*
shape
:@@*
dtype0*
	container *
shared_name *
_output_shapes

:@@
х
"training/RMSprop/Variable_2/AssignAssigntraining/RMSprop/Variable_2training/RMSprop/Const_2*
T0*
validate_shape(*
use_locking(*.
_class$
" loc:@training/RMSprop/Variable_2*
_output_shapes

:@@
Ђ
 training/RMSprop/Variable_2/readIdentitytraining/RMSprop/Variable_2*
T0*.
_class$
" loc:@training/RMSprop/Variable_2*
_output_shapes

:@@
e
training/RMSprop/Const_3Const*
valueB@*    *
dtype0*
_output_shapes
:@

training/RMSprop/Variable_3
VariableV2*
shape:@*
dtype0*
	container *
shared_name *
_output_shapes
:@
с
"training/RMSprop/Variable_3/AssignAssigntraining/RMSprop/Variable_3training/RMSprop/Const_3*
T0*
validate_shape(*
use_locking(*.
_class$
" loc:@training/RMSprop/Variable_3*
_output_shapes
:@

 training/RMSprop/Variable_3/readIdentitytraining/RMSprop/Variable_3*
T0*.
_class$
" loc:@training/RMSprop/Variable_3*
_output_shapes
:@
m
training/RMSprop/Const_4Const*
valueB@*    *
dtype0*
_output_shapes

:@

training/RMSprop/Variable_4
VariableV2*
shape
:@*
dtype0*
	container *
shared_name *
_output_shapes

:@
х
"training/RMSprop/Variable_4/AssignAssigntraining/RMSprop/Variable_4training/RMSprop/Const_4*
T0*
validate_shape(*
use_locking(*.
_class$
" loc:@training/RMSprop/Variable_4*
_output_shapes

:@
Ђ
 training/RMSprop/Variable_4/readIdentitytraining/RMSprop/Variable_4*
T0*.
_class$
" loc:@training/RMSprop/Variable_4*
_output_shapes

:@
e
training/RMSprop/Const_5Const*
valueB*    *
dtype0*
_output_shapes
:

training/RMSprop/Variable_5
VariableV2*
shape:*
dtype0*
	container *
shared_name *
_output_shapes
:
с
"training/RMSprop/Variable_5/AssignAssigntraining/RMSprop/Variable_5training/RMSprop/Const_5*
T0*
validate_shape(*
use_locking(*.
_class$
" loc:@training/RMSprop/Variable_5*
_output_shapes
:

 training/RMSprop/Variable_5/readIdentitytraining/RMSprop/Variable_5*
T0*.
_class$
" loc:@training/RMSprop/Variable_5*
_output_shapes
:
b
 training/RMSprop/AssignAdd/valueConst*
value	B	 R*
dtype0	*
_output_shapes
: 
И
training/RMSprop/AssignAdd	AssignAddRMSprop/iterations training/RMSprop/AssignAdd/value*
T0	*
use_locking( *%
_class
loc:@RMSprop/iterations*
_output_shapes
: 
v
training/RMSprop/mulMulRMSprop/rho/readtraining/RMSprop/Variable/read*
T0*
_output_shapes

:@
[
training/RMSprop/sub/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/RMSprop/subSubtraining/RMSprop/sub/xRMSprop/rho/read*
T0*
_output_shapes
: 

training/RMSprop/SquareSquare7training/RMSprop/gradients/dense_1/MatMul_grad/MatMul_1*
T0*
_output_shapes

:@
u
training/RMSprop/mul_1Multraining/RMSprop/subtraining/RMSprop/Square*
T0*
_output_shapes

:@
r
training/RMSprop/addAddtraining/RMSprop/multraining/RMSprop/mul_1*
T0*
_output_shapes

:@
в
training/RMSprop/AssignAssigntraining/RMSprop/Variabletraining/RMSprop/add*
T0*
validate_shape(*
use_locking(*,
_class"
 loc:@training/RMSprop/Variable*
_output_shapes

:@

training/RMSprop/mul_2MulRMSprop/lr/read7training/RMSprop/gradients/dense_1/MatMul_grad/MatMul_1*
T0*
_output_shapes

:@
]
training/RMSprop/Const_6Const*
valueB
 *    *
dtype0*
_output_shapes
: 
]
training/RMSprop/Const_7Const*
valueB
 *  *
dtype0*
_output_shapes
: 

&training/RMSprop/clip_by_value/MinimumMinimumtraining/RMSprop/addtraining/RMSprop/Const_7*
T0*
_output_shapes

:@

training/RMSprop/clip_by_valueMaximum&training/RMSprop/clip_by_value/Minimumtraining/RMSprop/Const_6*
T0*
_output_shapes

:@
f
training/RMSprop/SqrtSqrttraining/RMSprop/clip_by_value*
T0*
_output_shapes

:@
]
training/RMSprop/add_1/yConst*
valueB
 *wЬ+2*
dtype0*
_output_shapes
: 
w
training/RMSprop/add_1Addtraining/RMSprop/Sqrttraining/RMSprop/add_1/y*
T0*
_output_shapes

:@
|
training/RMSprop/truedivRealDivtraining/RMSprop/mul_2training/RMSprop/add_1*
T0*
_output_shapes

:@
u
training/RMSprop/sub_1Subdense_1/kernel/readtraining/RMSprop/truediv*
T0*
_output_shapes

:@
Р
training/RMSprop/Assign_1Assigndense_1/kerneltraining/RMSprop/sub_1*
T0*
validate_shape(*
use_locking(*!
_class
loc:@dense_1/kernel*
_output_shapes

:@
v
training/RMSprop/mul_3MulRMSprop/rho/read training/RMSprop/Variable_1/read*
T0*
_output_shapes
:@
]
training/RMSprop/sub_2/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
j
training/RMSprop/sub_2Subtraining/RMSprop/sub_2/xRMSprop/rho/read*
T0*
_output_shapes
: 

training/RMSprop/Square_1Square;training/RMSprop/gradients/dense_1/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:@
u
training/RMSprop/mul_4Multraining/RMSprop/sub_2training/RMSprop/Square_1*
T0*
_output_shapes
:@
r
training/RMSprop/add_2Addtraining/RMSprop/mul_3training/RMSprop/mul_4*
T0*
_output_shapes
:@
ж
training/RMSprop/Assign_2Assigntraining/RMSprop/Variable_1training/RMSprop/add_2*
T0*
validate_shape(*
use_locking(*.
_class$
" loc:@training/RMSprop/Variable_1*
_output_shapes
:@

training/RMSprop/mul_5MulRMSprop/lr/read;training/RMSprop/gradients/dense_1/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:@
]
training/RMSprop/Const_8Const*
valueB
 *    *
dtype0*
_output_shapes
: 
]
training/RMSprop/Const_9Const*
valueB
 *  *
dtype0*
_output_shapes
: 

(training/RMSprop/clip_by_value_1/MinimumMinimumtraining/RMSprop/add_2training/RMSprop/Const_9*
T0*
_output_shapes
:@

 training/RMSprop/clip_by_value_1Maximum(training/RMSprop/clip_by_value_1/Minimumtraining/RMSprop/Const_8*
T0*
_output_shapes
:@
f
training/RMSprop/Sqrt_1Sqrt training/RMSprop/clip_by_value_1*
T0*
_output_shapes
:@
]
training/RMSprop/add_3/yConst*
valueB
 *wЬ+2*
dtype0*
_output_shapes
: 
u
training/RMSprop/add_3Addtraining/RMSprop/Sqrt_1training/RMSprop/add_3/y*
T0*
_output_shapes
:@
z
training/RMSprop/truediv_1RealDivtraining/RMSprop/mul_5training/RMSprop/add_3*
T0*
_output_shapes
:@
q
training/RMSprop/sub_3Subdense_1/bias/readtraining/RMSprop/truediv_1*
T0*
_output_shapes
:@
И
training/RMSprop/Assign_3Assigndense_1/biastraining/RMSprop/sub_3*
T0*
validate_shape(*
use_locking(*
_class
loc:@dense_1/bias*
_output_shapes
:@
z
training/RMSprop/mul_6MulRMSprop/rho/read training/RMSprop/Variable_2/read*
T0*
_output_shapes

:@@
]
training/RMSprop/sub_4/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
j
training/RMSprop/sub_4Subtraining/RMSprop/sub_4/xRMSprop/rho/read*
T0*
_output_shapes
: 

training/RMSprop/Square_2Square7training/RMSprop/gradients/dense_2/MatMul_grad/MatMul_1*
T0*
_output_shapes

:@@
y
training/RMSprop/mul_7Multraining/RMSprop/sub_4training/RMSprop/Square_2*
T0*
_output_shapes

:@@
v
training/RMSprop/add_4Addtraining/RMSprop/mul_6training/RMSprop/mul_7*
T0*
_output_shapes

:@@
к
training/RMSprop/Assign_4Assigntraining/RMSprop/Variable_2training/RMSprop/add_4*
T0*
validate_shape(*
use_locking(*.
_class$
" loc:@training/RMSprop/Variable_2*
_output_shapes

:@@

training/RMSprop/mul_8MulRMSprop/lr/read7training/RMSprop/gradients/dense_2/MatMul_grad/MatMul_1*
T0*
_output_shapes

:@@
^
training/RMSprop/Const_10Const*
valueB
 *    *
dtype0*
_output_shapes
: 
^
training/RMSprop/Const_11Const*
valueB
 *  *
dtype0*
_output_shapes
: 

(training/RMSprop/clip_by_value_2/MinimumMinimumtraining/RMSprop/add_4training/RMSprop/Const_11*
T0*
_output_shapes

:@@

 training/RMSprop/clip_by_value_2Maximum(training/RMSprop/clip_by_value_2/Minimumtraining/RMSprop/Const_10*
T0*
_output_shapes

:@@
j
training/RMSprop/Sqrt_2Sqrt training/RMSprop/clip_by_value_2*
T0*
_output_shapes

:@@
]
training/RMSprop/add_5/yConst*
valueB
 *wЬ+2*
dtype0*
_output_shapes
: 
y
training/RMSprop/add_5Addtraining/RMSprop/Sqrt_2training/RMSprop/add_5/y*
T0*
_output_shapes

:@@
~
training/RMSprop/truediv_2RealDivtraining/RMSprop/mul_8training/RMSprop/add_5*
T0*
_output_shapes

:@@
w
training/RMSprop/sub_5Subdense_2/kernel/readtraining/RMSprop/truediv_2*
T0*
_output_shapes

:@@
Р
training/RMSprop/Assign_5Assigndense_2/kerneltraining/RMSprop/sub_5*
T0*
validate_shape(*
use_locking(*!
_class
loc:@dense_2/kernel*
_output_shapes

:@@
v
training/RMSprop/mul_9MulRMSprop/rho/read training/RMSprop/Variable_3/read*
T0*
_output_shapes
:@
]
training/RMSprop/sub_6/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
j
training/RMSprop/sub_6Subtraining/RMSprop/sub_6/xRMSprop/rho/read*
T0*
_output_shapes
: 

training/RMSprop/Square_3Square;training/RMSprop/gradients/dense_2/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:@
v
training/RMSprop/mul_10Multraining/RMSprop/sub_6training/RMSprop/Square_3*
T0*
_output_shapes
:@
s
training/RMSprop/add_6Addtraining/RMSprop/mul_9training/RMSprop/mul_10*
T0*
_output_shapes
:@
ж
training/RMSprop/Assign_6Assigntraining/RMSprop/Variable_3training/RMSprop/add_6*
T0*
validate_shape(*
use_locking(*.
_class$
" loc:@training/RMSprop/Variable_3*
_output_shapes
:@

training/RMSprop/mul_11MulRMSprop/lr/read;training/RMSprop/gradients/dense_2/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:@
^
training/RMSprop/Const_12Const*
valueB
 *    *
dtype0*
_output_shapes
: 
^
training/RMSprop/Const_13Const*
valueB
 *  *
dtype0*
_output_shapes
: 

(training/RMSprop/clip_by_value_3/MinimumMinimumtraining/RMSprop/add_6training/RMSprop/Const_13*
T0*
_output_shapes
:@

 training/RMSprop/clip_by_value_3Maximum(training/RMSprop/clip_by_value_3/Minimumtraining/RMSprop/Const_12*
T0*
_output_shapes
:@
f
training/RMSprop/Sqrt_3Sqrt training/RMSprop/clip_by_value_3*
T0*
_output_shapes
:@
]
training/RMSprop/add_7/yConst*
valueB
 *wЬ+2*
dtype0*
_output_shapes
: 
u
training/RMSprop/add_7Addtraining/RMSprop/Sqrt_3training/RMSprop/add_7/y*
T0*
_output_shapes
:@
{
training/RMSprop/truediv_3RealDivtraining/RMSprop/mul_11training/RMSprop/add_7*
T0*
_output_shapes
:@
q
training/RMSprop/sub_7Subdense_2/bias/readtraining/RMSprop/truediv_3*
T0*
_output_shapes
:@
И
training/RMSprop/Assign_7Assigndense_2/biastraining/RMSprop/sub_7*
T0*
validate_shape(*
use_locking(*
_class
loc:@dense_2/bias*
_output_shapes
:@
{
training/RMSprop/mul_12MulRMSprop/rho/read training/RMSprop/Variable_4/read*
T0*
_output_shapes

:@
]
training/RMSprop/sub_8/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
j
training/RMSprop/sub_8Subtraining/RMSprop/sub_8/xRMSprop/rho/read*
T0*
_output_shapes
: 

training/RMSprop/Square_4Square7training/RMSprop/gradients/dense_3/MatMul_grad/MatMul_1*
T0*
_output_shapes

:@
z
training/RMSprop/mul_13Multraining/RMSprop/sub_8training/RMSprop/Square_4*
T0*
_output_shapes

:@
x
training/RMSprop/add_8Addtraining/RMSprop/mul_12training/RMSprop/mul_13*
T0*
_output_shapes

:@
к
training/RMSprop/Assign_8Assigntraining/RMSprop/Variable_4training/RMSprop/add_8*
T0*
validate_shape(*
use_locking(*.
_class$
" loc:@training/RMSprop/Variable_4*
_output_shapes

:@

training/RMSprop/mul_14MulRMSprop/lr/read7training/RMSprop/gradients/dense_3/MatMul_grad/MatMul_1*
T0*
_output_shapes

:@
^
training/RMSprop/Const_14Const*
valueB
 *    *
dtype0*
_output_shapes
: 
^
training/RMSprop/Const_15Const*
valueB
 *  *
dtype0*
_output_shapes
: 

(training/RMSprop/clip_by_value_4/MinimumMinimumtraining/RMSprop/add_8training/RMSprop/Const_15*
T0*
_output_shapes

:@

 training/RMSprop/clip_by_value_4Maximum(training/RMSprop/clip_by_value_4/Minimumtraining/RMSprop/Const_14*
T0*
_output_shapes

:@
j
training/RMSprop/Sqrt_4Sqrt training/RMSprop/clip_by_value_4*
T0*
_output_shapes

:@
]
training/RMSprop/add_9/yConst*
valueB
 *wЬ+2*
dtype0*
_output_shapes
: 
y
training/RMSprop/add_9Addtraining/RMSprop/Sqrt_4training/RMSprop/add_9/y*
T0*
_output_shapes

:@

training/RMSprop/truediv_4RealDivtraining/RMSprop/mul_14training/RMSprop/add_9*
T0*
_output_shapes

:@
w
training/RMSprop/sub_9Subdense_3/kernel/readtraining/RMSprop/truediv_4*
T0*
_output_shapes

:@
Р
training/RMSprop/Assign_9Assigndense_3/kerneltraining/RMSprop/sub_9*
T0*
validate_shape(*
use_locking(*!
_class
loc:@dense_3/kernel*
_output_shapes

:@
w
training/RMSprop/mul_15MulRMSprop/rho/read training/RMSprop/Variable_5/read*
T0*
_output_shapes
:
^
training/RMSprop/sub_10/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
l
training/RMSprop/sub_10Subtraining/RMSprop/sub_10/xRMSprop/rho/read*
T0*
_output_shapes
: 

training/RMSprop/Square_5Square;training/RMSprop/gradients/dense_3/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:
w
training/RMSprop/mul_16Multraining/RMSprop/sub_10training/RMSprop/Square_5*
T0*
_output_shapes
:
u
training/RMSprop/add_10Addtraining/RMSprop/mul_15training/RMSprop/mul_16*
T0*
_output_shapes
:
и
training/RMSprop/Assign_10Assigntraining/RMSprop/Variable_5training/RMSprop/add_10*
T0*
validate_shape(*
use_locking(*.
_class$
" loc:@training/RMSprop/Variable_5*
_output_shapes
:

training/RMSprop/mul_17MulRMSprop/lr/read;training/RMSprop/gradients/dense_3/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:
^
training/RMSprop/Const_16Const*
valueB
 *    *
dtype0*
_output_shapes
: 
^
training/RMSprop/Const_17Const*
valueB
 *  *
dtype0*
_output_shapes
: 

(training/RMSprop/clip_by_value_5/MinimumMinimumtraining/RMSprop/add_10training/RMSprop/Const_17*
T0*
_output_shapes
:

 training/RMSprop/clip_by_value_5Maximum(training/RMSprop/clip_by_value_5/Minimumtraining/RMSprop/Const_16*
T0*
_output_shapes
:
f
training/RMSprop/Sqrt_5Sqrt training/RMSprop/clip_by_value_5*
T0*
_output_shapes
:
^
training/RMSprop/add_11/yConst*
valueB
 *wЬ+2*
dtype0*
_output_shapes
: 
w
training/RMSprop/add_11Addtraining/RMSprop/Sqrt_5training/RMSprop/add_11/y*
T0*
_output_shapes
:
|
training/RMSprop/truediv_5RealDivtraining/RMSprop/mul_17training/RMSprop/add_11*
T0*
_output_shapes
:
r
training/RMSprop/sub_11Subdense_3/bias/readtraining/RMSprop/truediv_5*
T0*
_output_shapes
:
К
training/RMSprop/Assign_11Assigndense_3/biastraining/RMSprop/sub_11*
T0*
validate_shape(*
use_locking(*
_class
loc:@dense_3/bias*
_output_shapes
:
И
training/group_depsNoOp	^loss/mul#^metrics/mean_absolute_error/Mean_1^training/RMSprop/AssignAdd^training/RMSprop/Assign^training/RMSprop/Assign_1^training/RMSprop/Assign_2^training/RMSprop/Assign_3^training/RMSprop/Assign_4^training/RMSprop/Assign_5^training/RMSprop/Assign_6^training/RMSprop/Assign_7^training/RMSprop/Assign_8^training/RMSprop/Assign_9^training/RMSprop/Assign_10^training/RMSprop/Assign_11
B

group_depsNoOp	^loss/mul#^metrics/mean_absolute_error/Mean_1

IsVariableInitializedIsVariableInitializeddense_1/kernel*
dtype0*!
_class
loc:@dense_1/kernel*
_output_shapes
: 

IsVariableInitialized_1IsVariableInitializeddense_1/bias*
dtype0*
_class
loc:@dense_1/bias*
_output_shapes
: 

IsVariableInitialized_2IsVariableInitializeddense_2/kernel*
dtype0*!
_class
loc:@dense_2/kernel*
_output_shapes
: 

IsVariableInitialized_3IsVariableInitializeddense_2/bias*
dtype0*
_class
loc:@dense_2/bias*
_output_shapes
: 

IsVariableInitialized_4IsVariableInitializeddense_3/kernel*
dtype0*!
_class
loc:@dense_3/kernel*
_output_shapes
: 

IsVariableInitialized_5IsVariableInitializeddense_3/bias*
dtype0*
_class
loc:@dense_3/bias*
_output_shapes
: 

IsVariableInitialized_6IsVariableInitialized
RMSprop/lr*
dtype0*
_class
loc:@RMSprop/lr*
_output_shapes
: 

IsVariableInitialized_7IsVariableInitializedRMSprop/rho*
dtype0*
_class
loc:@RMSprop/rho*
_output_shapes
: 

IsVariableInitialized_8IsVariableInitializedRMSprop/decay*
dtype0* 
_class
loc:@RMSprop/decay*
_output_shapes
: 

IsVariableInitialized_9IsVariableInitializedRMSprop/iterations*
dtype0	*%
_class
loc:@RMSprop/iterations*
_output_shapes
: 

IsVariableInitialized_10IsVariableInitializedtraining/RMSprop/Variable*
dtype0*,
_class"
 loc:@training/RMSprop/Variable*
_output_shapes
: 
Ѓ
IsVariableInitialized_11IsVariableInitializedtraining/RMSprop/Variable_1*
dtype0*.
_class$
" loc:@training/RMSprop/Variable_1*
_output_shapes
: 
Ѓ
IsVariableInitialized_12IsVariableInitializedtraining/RMSprop/Variable_2*
dtype0*.
_class$
" loc:@training/RMSprop/Variable_2*
_output_shapes
: 
Ѓ
IsVariableInitialized_13IsVariableInitializedtraining/RMSprop/Variable_3*
dtype0*.
_class$
" loc:@training/RMSprop/Variable_3*
_output_shapes
: 
Ѓ
IsVariableInitialized_14IsVariableInitializedtraining/RMSprop/Variable_4*
dtype0*.
_class$
" loc:@training/RMSprop/Variable_4*
_output_shapes
: 
Ѓ
IsVariableInitialized_15IsVariableInitializedtraining/RMSprop/Variable_5*
dtype0*.
_class$
" loc:@training/RMSprop/Variable_5*
_output_shapes
: 
Ю
initNoOp^dense_1/kernel/Assign^dense_1/bias/Assign^dense_2/kernel/Assign^dense_2/bias/Assign^dense_3/kernel/Assign^dense_3/bias/Assign^RMSprop/lr/Assign^RMSprop/rho/Assign^RMSprop/decay/Assign^RMSprop/iterations/Assign!^training/RMSprop/Variable/Assign#^training/RMSprop/Variable_1/Assign#^training/RMSprop/Variable_2/Assign#^training/RMSprop/Variable_3/Assign#^training/RMSprop/Variable_4/Assign#^training/RMSprop/Variable_5/Assign
e
dense_1/kernel_0/tagConst*!
valueB Bdense_1/kernel_0*
dtype0*
_output_shapes
: 
p
dense_1/kernel_0HistogramSummarydense_1/kernel_0/tagdense_1/kernel/read*
T0*
_output_shapes
: 
a
dense_1/bias_0/tagConst*
valueB Bdense_1/bias_0*
dtype0*
_output_shapes
: 
j
dense_1/bias_0HistogramSummarydense_1/bias_0/tagdense_1/bias/read*
T0*
_output_shapes
: 
[
dense_1_out/tagConst*
valueB Bdense_1_out*
dtype0*
_output_shapes
: 
_
dense_1_outHistogramSummarydense_1_out/tagdense_1/Relu*
T0*
_output_shapes
: 
e
dense_2/kernel_0/tagConst*!
valueB Bdense_2/kernel_0*
dtype0*
_output_shapes
: 
p
dense_2/kernel_0HistogramSummarydense_2/kernel_0/tagdense_2/kernel/read*
T0*
_output_shapes
: 
a
dense_2/bias_0/tagConst*
valueB Bdense_2/bias_0*
dtype0*
_output_shapes
: 
j
dense_2/bias_0HistogramSummarydense_2/bias_0/tagdense_2/bias/read*
T0*
_output_shapes
: 
[
dense_2_out/tagConst*
valueB Bdense_2_out*
dtype0*
_output_shapes
: 
_
dense_2_outHistogramSummarydense_2_out/tagdense_2/Relu*
T0*
_output_shapes
: 
e
dense_3/kernel_0/tagConst*!
valueB Bdense_3/kernel_0*
dtype0*
_output_shapes
: 
p
dense_3/kernel_0HistogramSummarydense_3/kernel_0/tagdense_3/kernel/read*
T0*
_output_shapes
: 
a
dense_3/bias_0/tagConst*
valueB Bdense_3/bias_0*
dtype0*
_output_shapes
: 
j
dense_3/bias_0HistogramSummarydense_3/bias_0/tagdense_3/bias/read*
T0*
_output_shapes
: 
[
dense_3_out/tagConst*
valueB Bdense_3_out*
dtype0*
_output_shapes
: 
b
dense_3_outHistogramSummarydense_3_out/tagdense_3/BiasAdd*
T0*
_output_shapes
: 
а
Merge/MergeSummaryMergeSummarydense_1/kernel_0dense_1/bias_0dense_1_outdense_2/kernel_0dense_2/bias_0dense_2_outdense_3/kernel_0dense_3/bias_0dense_3_out*
N	*
_output_shapes
: 
p
dense_4_inputPlaceholder*
dtype0*
shape:џџџџџџџџџ*'
_output_shapes
:џџџџџџџџџ
m
dense_4/random_uniform/shapeConst*
valueB"   @   *
dtype0*
_output_shapes
:
_
dense_4/random_uniform/minConst*
valueB
 *!ьО*
dtype0*
_output_shapes
: 
_
dense_4/random_uniform/maxConst*
valueB
 *!ь>*
dtype0*
_output_shapes
: 
Ї
$dense_4/random_uniform/RandomUniformRandomUniformdense_4/random_uniform/shape*
seedБџх)*
seed2ЈшD*
dtype0*
T0*
_output_shapes

:@
z
dense_4/random_uniform/subSubdense_4/random_uniform/maxdense_4/random_uniform/min*
T0*
_output_shapes
: 

dense_4/random_uniform/mulMul$dense_4/random_uniform/RandomUniformdense_4/random_uniform/sub*
T0*
_output_shapes

:@
~
dense_4/random_uniformAdddense_4/random_uniform/muldense_4/random_uniform/min*
T0*
_output_shapes

:@

dense_4/kernel
VariableV2*
shape
:@*
dtype0*
	container *
shared_name *
_output_shapes

:@
М
dense_4/kernel/AssignAssigndense_4/kerneldense_4/random_uniform*
T0*
validate_shape(*
use_locking(*!
_class
loc:@dense_4/kernel*
_output_shapes

:@
{
dense_4/kernel/readIdentitydense_4/kernel*
T0*!
_class
loc:@dense_4/kernel*
_output_shapes

:@
Z
dense_4/ConstConst*
valueB@*    *
dtype0*
_output_shapes
:@
x
dense_4/bias
VariableV2*
shape:@*
dtype0*
	container *
shared_name *
_output_shapes
:@
Љ
dense_4/bias/AssignAssigndense_4/biasdense_4/Const*
T0*
validate_shape(*
use_locking(*
_class
loc:@dense_4/bias*
_output_shapes
:@
q
dense_4/bias/readIdentitydense_4/bias*
T0*
_class
loc:@dense_4/bias*
_output_shapes
:@

dense_4/MatMulMatMuldense_4_inputdense_4/kernel/read*
transpose_a( *
transpose_b( *
T0*'
_output_shapes
:џџџџџџџџџ@

dense_4/BiasAddBiasAdddense_4/MatMuldense_4/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:џџџџџџџџџ@
W
dense_4/ReluReludense_4/BiasAdd*
T0*'
_output_shapes
:џџџџџџџџџ@
m
dense_5/random_uniform/shapeConst*
valueB"@   @   *
dtype0*
_output_shapes
:
_
dense_5/random_uniform/minConst*
valueB
 *зГ]О*
dtype0*
_output_shapes
: 
_
dense_5/random_uniform/maxConst*
valueB
 *зГ]>*
dtype0*
_output_shapes
: 
Ї
$dense_5/random_uniform/RandomUniformRandomUniformdense_5/random_uniform/shape*
seedБџх)*
seed2їн *
dtype0*
T0*
_output_shapes

:@@
z
dense_5/random_uniform/subSubdense_5/random_uniform/maxdense_5/random_uniform/min*
T0*
_output_shapes
: 

dense_5/random_uniform/mulMul$dense_5/random_uniform/RandomUniformdense_5/random_uniform/sub*
T0*
_output_shapes

:@@
~
dense_5/random_uniformAdddense_5/random_uniform/muldense_5/random_uniform/min*
T0*
_output_shapes

:@@

dense_5/kernel
VariableV2*
shape
:@@*
dtype0*
	container *
shared_name *
_output_shapes

:@@
М
dense_5/kernel/AssignAssigndense_5/kerneldense_5/random_uniform*
T0*
validate_shape(*
use_locking(*!
_class
loc:@dense_5/kernel*
_output_shapes

:@@
{
dense_5/kernel/readIdentitydense_5/kernel*
T0*!
_class
loc:@dense_5/kernel*
_output_shapes

:@@
Z
dense_5/ConstConst*
valueB@*    *
dtype0*
_output_shapes
:@
x
dense_5/bias
VariableV2*
shape:@*
dtype0*
	container *
shared_name *
_output_shapes
:@
Љ
dense_5/bias/AssignAssigndense_5/biasdense_5/Const*
T0*
validate_shape(*
use_locking(*
_class
loc:@dense_5/bias*
_output_shapes
:@
q
dense_5/bias/readIdentitydense_5/bias*
T0*
_class
loc:@dense_5/bias*
_output_shapes
:@

dense_5/MatMulMatMuldense_4/Reludense_5/kernel/read*
transpose_a( *
transpose_b( *
T0*'
_output_shapes
:џџџџџџџџџ@

dense_5/BiasAddBiasAdddense_5/MatMuldense_5/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:џџџџџџџџџ@
W
dense_5/ReluReludense_5/BiasAdd*
T0*'
_output_shapes
:џџџџџџџџџ@
m
dense_6/random_uniform/shapeConst*
valueB"@      *
dtype0*
_output_shapes
:
_
dense_6/random_uniform/minConst*
valueB
 *О*
dtype0*
_output_shapes
: 
_
dense_6/random_uniform/maxConst*
valueB
 *>*
dtype0*
_output_shapes
: 
Ј
$dense_6/random_uniform/RandomUniformRandomUniformdense_6/random_uniform/shape*
seedБџх)*
seed2мњ*
dtype0*
T0*
_output_shapes

:@
z
dense_6/random_uniform/subSubdense_6/random_uniform/maxdense_6/random_uniform/min*
T0*
_output_shapes
: 

dense_6/random_uniform/mulMul$dense_6/random_uniform/RandomUniformdense_6/random_uniform/sub*
T0*
_output_shapes

:@
~
dense_6/random_uniformAdddense_6/random_uniform/muldense_6/random_uniform/min*
T0*
_output_shapes

:@

dense_6/kernel
VariableV2*
shape
:@*
dtype0*
	container *
shared_name *
_output_shapes

:@
М
dense_6/kernel/AssignAssigndense_6/kerneldense_6/random_uniform*
T0*
validate_shape(*
use_locking(*!
_class
loc:@dense_6/kernel*
_output_shapes

:@
{
dense_6/kernel/readIdentitydense_6/kernel*
T0*!
_class
loc:@dense_6/kernel*
_output_shapes

:@
Z
dense_6/ConstConst*
valueB*    *
dtype0*
_output_shapes
:
x
dense_6/bias
VariableV2*
shape:*
dtype0*
	container *
shared_name *
_output_shapes
:
Љ
dense_6/bias/AssignAssigndense_6/biasdense_6/Const*
T0*
validate_shape(*
use_locking(*
_class
loc:@dense_6/bias*
_output_shapes
:
q
dense_6/bias/readIdentitydense_6/bias*
T0*
_class
loc:@dense_6/bias*
_output_shapes
:

dense_6/MatMulMatMuldense_5/Reludense_6/kernel/read*
transpose_a( *
transpose_b( *
T0*'
_output_shapes
:џџџџџџџџџ

dense_6/BiasAddBiasAdddense_6/MatMuldense_6/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:џџџџџџџџџ
_
RMSprop_1/lr/initial_valueConst*
valueB
 *o:*
dtype0*
_output_shapes
: 
p
RMSprop_1/lr
VariableV2*
shape: *
dtype0*
	container *
shared_name *
_output_shapes
: 
В
RMSprop_1/lr/AssignAssignRMSprop_1/lrRMSprop_1/lr/initial_value*
T0*
validate_shape(*
use_locking(*
_class
loc:@RMSprop_1/lr*
_output_shapes
: 
m
RMSprop_1/lr/readIdentityRMSprop_1/lr*
T0*
_class
loc:@RMSprop_1/lr*
_output_shapes
: 
`
RMSprop_1/rho/initial_valueConst*
valueB
 *fff?*
dtype0*
_output_shapes
: 
q
RMSprop_1/rho
VariableV2*
shape: *
dtype0*
	container *
shared_name *
_output_shapes
: 
Ж
RMSprop_1/rho/AssignAssignRMSprop_1/rhoRMSprop_1/rho/initial_value*
T0*
validate_shape(*
use_locking(* 
_class
loc:@RMSprop_1/rho*
_output_shapes
: 
p
RMSprop_1/rho/readIdentityRMSprop_1/rho*
T0* 
_class
loc:@RMSprop_1/rho*
_output_shapes
: 
b
RMSprop_1/decay/initial_valueConst*
valueB
 *    *
dtype0*
_output_shapes
: 
s
RMSprop_1/decay
VariableV2*
shape: *
dtype0*
	container *
shared_name *
_output_shapes
: 
О
RMSprop_1/decay/AssignAssignRMSprop_1/decayRMSprop_1/decay/initial_value*
T0*
validate_shape(*
use_locking(*"
_class
loc:@RMSprop_1/decay*
_output_shapes
: 
v
RMSprop_1/decay/readIdentityRMSprop_1/decay*
T0*"
_class
loc:@RMSprop_1/decay*
_output_shapes
: 
d
"RMSprop_1/iterations/initial_valueConst*
value	B	 R *
dtype0	*
_output_shapes
: 
x
RMSprop_1/iterations
VariableV2*
shape: *
dtype0	*
	container *
shared_name *
_output_shapes
: 
в
RMSprop_1/iterations/AssignAssignRMSprop_1/iterations"RMSprop_1/iterations/initial_value*
T0	*
validate_shape(*
use_locking(*'
_class
loc:@RMSprop_1/iterations*
_output_shapes
: 

RMSprop_1/iterations/readIdentityRMSprop_1/iterations*
T0	*'
_class
loc:@RMSprop_1/iterations*
_output_shapes
: 

dense_6_targetPlaceholder*
dtype0*%
shape:џџџџџџџџџџџџџџџџџџ*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
q
dense_6_sample_weightsPlaceholder*
dtype0*
shape:џџџџџџџџџ*#
_output_shapes
:џџџџџџџџџ
z
loss_1/dense_6_loss/subSubdense_6/BiasAdddense_6_target*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
x
loss_1/dense_6_loss/SquareSquareloss_1/dense_6_loss/sub*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
u
*loss_1/dense_6_loss/Mean/reduction_indicesConst*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 
Г
loss_1/dense_6_loss/MeanMeanloss_1/dense_6_loss/Square*loss_1/dense_6_loss/Mean/reduction_indices*
	keep_dims( *
T0*

Tidx0*#
_output_shapes
:џџџџџџџџџ
o
,loss_1/dense_6_loss/Mean_1/reduction_indicesConst*
valueB *
dtype0*
_output_shapes
: 
Е
loss_1/dense_6_loss/Mean_1Meanloss_1/dense_6_loss/Mean,loss_1/dense_6_loss/Mean_1/reduction_indices*
	keep_dims( *
T0*

Tidx0*#
_output_shapes
:џџџџџџџџџ

loss_1/dense_6_loss/mulMulloss_1/dense_6_loss/Mean_1dense_6_sample_weights*
T0*#
_output_shapes
:џџџџџџџџџ
c
loss_1/dense_6_loss/NotEqual/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 

loss_1/dense_6_loss/NotEqualNotEqualdense_6_sample_weightsloss_1/dense_6_loss/NotEqual/y*
T0*#
_output_shapes
:џџџџџџџџџ
{
loss_1/dense_6_loss/CastCastloss_1/dense_6_loss/NotEqual*

SrcT0
*

DstT0*#
_output_shapes
:џџџџџџџџџ
c
loss_1/dense_6_loss/ConstConst*
valueB: *
dtype0*
_output_shapes
:

loss_1/dense_6_loss/Mean_2Meanloss_1/dense_6_loss/Castloss_1/dense_6_loss/Const*
	keep_dims( *
T0*

Tidx0*
_output_shapes
: 

loss_1/dense_6_loss/truedivRealDivloss_1/dense_6_loss/mulloss_1/dense_6_loss/Mean_2*
T0*#
_output_shapes
:џџџџџџџџџ
e
loss_1/dense_6_loss/Const_1Const*
valueB: *
dtype0*
_output_shapes
:

loss_1/dense_6_loss/Mean_3Meanloss_1/dense_6_loss/truedivloss_1/dense_6_loss/Const_1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
: 
Q
loss_1/mul/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
\

loss_1/mulMulloss_1/mul/xloss_1/dense_6_loss/Mean_3*
T0*
_output_shapes
: 

!metrics_1/mean_absolute_error/subSubdense_6/BiasAdddense_6_target*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ

!metrics_1/mean_absolute_error/AbsAbs!metrics_1/mean_absolute_error/sub*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ

4metrics_1/mean_absolute_error/Mean/reduction_indicesConst*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 
Ю
"metrics_1/mean_absolute_error/MeanMean!metrics_1/mean_absolute_error/Abs4metrics_1/mean_absolute_error/Mean/reduction_indices*
	keep_dims( *
T0*

Tidx0*#
_output_shapes
:џџџџџџџџџ
m
#metrics_1/mean_absolute_error/ConstConst*
valueB: *
dtype0*
_output_shapes
:
Г
$metrics_1/mean_absolute_error/Mean_1Mean"metrics_1/mean_absolute_error/Mean#metrics_1/mean_absolute_error/Const*
	keep_dims( *
T0*

Tidx0*
_output_shapes
: 

"training_1/RMSprop/gradients/ShapeConst*
valueB *
dtype0*
_class
loc:@loss_1/mul*
_output_shapes
: 

&training_1/RMSprop/gradients/grad_ys_0Const*
valueB
 *  ?*
dtype0*
_class
loc:@loss_1/mul*
_output_shapes
: 
Е
!training_1/RMSprop/gradients/FillFill"training_1/RMSprop/gradients/Shape&training_1/RMSprop/gradients/grad_ys_0*
T0*
_class
loc:@loss_1/mul*
_output_shapes
: 
Ж
0training_1/RMSprop/gradients/loss_1/mul_grad/MulMul!training_1/RMSprop/gradients/Fillloss_1/dense_6_loss/Mean_3*
T0*
_class
loc:@loss_1/mul*
_output_shapes
: 
Њ
2training_1/RMSprop/gradients/loss_1/mul_grad/Mul_1Mul!training_1/RMSprop/gradients/Fillloss_1/mul/x*
T0*
_class
loc:@loss_1/mul*
_output_shapes
: 
У
Jtraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_3_grad/Reshape/shapeConst*
valueB:*
dtype0*-
_class#
!loc:@loss_1/dense_6_loss/Mean_3*
_output_shapes
:
Б
Dtraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_3_grad/ReshapeReshape2training_1/RMSprop/gradients/loss_1/mul_grad/Mul_1Jtraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_3_grad/Reshape/shape*
T0*
Tshape0*-
_class#
!loc:@loss_1/dense_6_loss/Mean_3*
_output_shapes
:
Ь
Btraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_3_grad/ShapeShapeloss_1/dense_6_loss/truediv*
T0*
out_type0*-
_class#
!loc:@loss_1/dense_6_loss/Mean_3*
_output_shapes
:
Т
Atraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_3_grad/TileTileDtraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_3_grad/ReshapeBtraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_3_grad/Shape*
T0*

Tmultiples0*-
_class#
!loc:@loss_1/dense_6_loss/Mean_3*#
_output_shapes
:џџџџџџџџџ
Ю
Dtraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_3_grad/Shape_1Shapeloss_1/dense_6_loss/truediv*
T0*
out_type0*-
_class#
!loc:@loss_1/dense_6_loss/Mean_3*
_output_shapes
:
Ж
Dtraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_3_grad/Shape_2Const*
valueB *
dtype0*-
_class#
!loc:@loss_1/dense_6_loss/Mean_3*
_output_shapes
: 
Л
Btraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_3_grad/ConstConst*
valueB: *
dtype0*-
_class#
!loc:@loss_1/dense_6_loss/Mean_3*
_output_shapes
:
Р
Atraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_3_grad/ProdProdDtraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_3_grad/Shape_1Btraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_3_grad/Const*
	keep_dims( *
T0*

Tidx0*-
_class#
!loc:@loss_1/dense_6_loss/Mean_3*
_output_shapes
: 
Н
Dtraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_3_grad/Const_1Const*
valueB: *
dtype0*-
_class#
!loc:@loss_1/dense_6_loss/Mean_3*
_output_shapes
:
Ф
Ctraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_3_grad/Prod_1ProdDtraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_3_grad/Shape_2Dtraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_3_grad/Const_1*
	keep_dims( *
T0*

Tidx0*-
_class#
!loc:@loss_1/dense_6_loss/Mean_3*
_output_shapes
: 
З
Ftraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_3_grad/Maximum/yConst*
value	B :*
dtype0*-
_class#
!loc:@loss_1/dense_6_loss/Mean_3*
_output_shapes
: 
Ќ
Dtraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_3_grad/MaximumMaximumCtraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_3_grad/Prod_1Ftraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_3_grad/Maximum/y*
T0*-
_class#
!loc:@loss_1/dense_6_loss/Mean_3*
_output_shapes
: 
Њ
Etraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_3_grad/floordivFloorDivAtraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_3_grad/ProdDtraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_3_grad/Maximum*
T0*-
_class#
!loc:@loss_1/dense_6_loss/Mean_3*
_output_shapes
: 
я
Atraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_3_grad/CastCastEtraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_3_grad/floordiv*

SrcT0*

DstT0*-
_class#
!loc:@loss_1/dense_6_loss/Mean_3*
_output_shapes
: 
В
Dtraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_3_grad/truedivRealDivAtraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_3_grad/TileAtraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_3_grad/Cast*
T0*-
_class#
!loc:@loss_1/dense_6_loss/Mean_3*#
_output_shapes
:џџџџџџџџџ
Ъ
Ctraining_1/RMSprop/gradients/loss_1/dense_6_loss/truediv_grad/ShapeShapeloss_1/dense_6_loss/mul*
T0*
out_type0*.
_class$
" loc:@loss_1/dense_6_loss/truediv*
_output_shapes
:
И
Etraining_1/RMSprop/gradients/loss_1/dense_6_loss/truediv_grad/Shape_1Const*
valueB *
dtype0*.
_class$
" loc:@loss_1/dense_6_loss/truediv*
_output_shapes
: 
х
Straining_1/RMSprop/gradients/loss_1/dense_6_loss/truediv_grad/BroadcastGradientArgsBroadcastGradientArgsCtraining_1/RMSprop/gradients/loss_1/dense_6_loss/truediv_grad/ShapeEtraining_1/RMSprop/gradients/loss_1/dense_6_loss/truediv_grad/Shape_1*
T0*.
_class$
" loc:@loss_1/dense_6_loss/truediv*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ

Etraining_1/RMSprop/gradients/loss_1/dense_6_loss/truediv_grad/RealDivRealDivDtraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_3_grad/truedivloss_1/dense_6_loss/Mean_2*
T0*.
_class$
" loc:@loss_1/dense_6_loss/truediv*#
_output_shapes
:џџџџџџџџџ
д
Atraining_1/RMSprop/gradients/loss_1/dense_6_loss/truediv_grad/SumSumEtraining_1/RMSprop/gradients/loss_1/dense_6_loss/truediv_grad/RealDivStraining_1/RMSprop/gradients/loss_1/dense_6_loss/truediv_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*.
_class$
" loc:@loss_1/dense_6_loss/truediv*
_output_shapes
:
Ф
Etraining_1/RMSprop/gradients/loss_1/dense_6_loss/truediv_grad/ReshapeReshapeAtraining_1/RMSprop/gradients/loss_1/dense_6_loss/truediv_grad/SumCtraining_1/RMSprop/gradients/loss_1/dense_6_loss/truediv_grad/Shape*
T0*
Tshape0*.
_class$
" loc:@loss_1/dense_6_loss/truediv*#
_output_shapes
:џџџџџџџџџ
П
Atraining_1/RMSprop/gradients/loss_1/dense_6_loss/truediv_grad/NegNegloss_1/dense_6_loss/mul*
T0*.
_class$
" loc:@loss_1/dense_6_loss/truediv*#
_output_shapes
:џџџџџџџџџ

Gtraining_1/RMSprop/gradients/loss_1/dense_6_loss/truediv_grad/RealDiv_1RealDivAtraining_1/RMSprop/gradients/loss_1/dense_6_loss/truediv_grad/Negloss_1/dense_6_loss/Mean_2*
T0*.
_class$
" loc:@loss_1/dense_6_loss/truediv*#
_output_shapes
:џџџџџџџџџ

Gtraining_1/RMSprop/gradients/loss_1/dense_6_loss/truediv_grad/RealDiv_2RealDivGtraining_1/RMSprop/gradients/loss_1/dense_6_loss/truediv_grad/RealDiv_1loss_1/dense_6_loss/Mean_2*
T0*.
_class$
" loc:@loss_1/dense_6_loss/truediv*#
_output_shapes
:џџџџџџџџџ
Е
Atraining_1/RMSprop/gradients/loss_1/dense_6_loss/truediv_grad/mulMulDtraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_3_grad/truedivGtraining_1/RMSprop/gradients/loss_1/dense_6_loss/truediv_grad/RealDiv_2*
T0*.
_class$
" loc:@loss_1/dense_6_loss/truediv*#
_output_shapes
:џџџџџџџџџ
д
Ctraining_1/RMSprop/gradients/loss_1/dense_6_loss/truediv_grad/Sum_1SumAtraining_1/RMSprop/gradients/loss_1/dense_6_loss/truediv_grad/mulUtraining_1/RMSprop/gradients/loss_1/dense_6_loss/truediv_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*.
_class$
" loc:@loss_1/dense_6_loss/truediv*
_output_shapes
:
Н
Gtraining_1/RMSprop/gradients/loss_1/dense_6_loss/truediv_grad/Reshape_1ReshapeCtraining_1/RMSprop/gradients/loss_1/dense_6_loss/truediv_grad/Sum_1Etraining_1/RMSprop/gradients/loss_1/dense_6_loss/truediv_grad/Shape_1*
T0*
Tshape0*.
_class$
" loc:@loss_1/dense_6_loss/truediv*
_output_shapes
: 
Х
?training_1/RMSprop/gradients/loss_1/dense_6_loss/mul_grad/ShapeShapeloss_1/dense_6_loss/Mean_1*
T0*
out_type0**
_class 
loc:@loss_1/dense_6_loss/mul*
_output_shapes
:
У
Atraining_1/RMSprop/gradients/loss_1/dense_6_loss/mul_grad/Shape_1Shapedense_6_sample_weights*
T0*
out_type0**
_class 
loc:@loss_1/dense_6_loss/mul*
_output_shapes
:
е
Otraining_1/RMSprop/gradients/loss_1/dense_6_loss/mul_grad/BroadcastGradientArgsBroadcastGradientArgs?training_1/RMSprop/gradients/loss_1/dense_6_loss/mul_grad/ShapeAtraining_1/RMSprop/gradients/loss_1/dense_6_loss/mul_grad/Shape_1*
T0**
_class 
loc:@loss_1/dense_6_loss/mul*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
§
=training_1/RMSprop/gradients/loss_1/dense_6_loss/mul_grad/mulMulEtraining_1/RMSprop/gradients/loss_1/dense_6_loss/truediv_grad/Reshapedense_6_sample_weights*
T0**
_class 
loc:@loss_1/dense_6_loss/mul*#
_output_shapes
:џџџџџџџџџ
Р
=training_1/RMSprop/gradients/loss_1/dense_6_loss/mul_grad/SumSum=training_1/RMSprop/gradients/loss_1/dense_6_loss/mul_grad/mulOtraining_1/RMSprop/gradients/loss_1/dense_6_loss/mul_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0**
_class 
loc:@loss_1/dense_6_loss/mul*
_output_shapes
:
Д
Atraining_1/RMSprop/gradients/loss_1/dense_6_loss/mul_grad/ReshapeReshape=training_1/RMSprop/gradients/loss_1/dense_6_loss/mul_grad/Sum?training_1/RMSprop/gradients/loss_1/dense_6_loss/mul_grad/Shape*
T0*
Tshape0**
_class 
loc:@loss_1/dense_6_loss/mul*#
_output_shapes
:џџџџџџџџџ

?training_1/RMSprop/gradients/loss_1/dense_6_loss/mul_grad/mul_1Mulloss_1/dense_6_loss/Mean_1Etraining_1/RMSprop/gradients/loss_1/dense_6_loss/truediv_grad/Reshape*
T0**
_class 
loc:@loss_1/dense_6_loss/mul*#
_output_shapes
:џџџџџџџџџ
Ц
?training_1/RMSprop/gradients/loss_1/dense_6_loss/mul_grad/Sum_1Sum?training_1/RMSprop/gradients/loss_1/dense_6_loss/mul_grad/mul_1Qtraining_1/RMSprop/gradients/loss_1/dense_6_loss/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0**
_class 
loc:@loss_1/dense_6_loss/mul*
_output_shapes
:
К
Ctraining_1/RMSprop/gradients/loss_1/dense_6_loss/mul_grad/Reshape_1Reshape?training_1/RMSprop/gradients/loss_1/dense_6_loss/mul_grad/Sum_1Atraining_1/RMSprop/gradients/loss_1/dense_6_loss/mul_grad/Shape_1*
T0*
Tshape0**
_class 
loc:@loss_1/dense_6_loss/mul*#
_output_shapes
:џџџџџџџџџ
Щ
Btraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_1_grad/ShapeShapeloss_1/dense_6_loss/Mean*
T0*
out_type0*-
_class#
!loc:@loss_1/dense_6_loss/Mean_1*
_output_shapes
:
В
Atraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_1_grad/SizeConst*
value	B :*
dtype0*-
_class#
!loc:@loss_1/dense_6_loss/Mean_1*
_output_shapes
: 

@training_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_1_grad/addAdd,loss_1/dense_6_loss/Mean_1/reduction_indicesAtraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_1_grad/Size*
T0*-
_class#
!loc:@loss_1/dense_6_loss/Mean_1*
_output_shapes
: 
Ѓ
@training_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_1_grad/modFloorMod@training_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_1_grad/addAtraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_1_grad/Size*
T0*-
_class#
!loc:@loss_1/dense_6_loss/Mean_1*
_output_shapes
: 
Н
Dtraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_1_grad/Shape_1Const*
valueB: *
dtype0*-
_class#
!loc:@loss_1/dense_6_loss/Mean_1*
_output_shapes
:
Й
Htraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_1_grad/range/startConst*
value	B : *
dtype0*-
_class#
!loc:@loss_1/dense_6_loss/Mean_1*
_output_shapes
: 
Й
Htraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_1_grad/range/deltaConst*
value	B :*
dtype0*-
_class#
!loc:@loss_1/dense_6_loss/Mean_1*
_output_shapes
: 
љ
Btraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_1_grad/rangeRangeHtraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_1_grad/range/startAtraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_1_grad/SizeHtraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_1_grad/range/delta*

Tidx0*-
_class#
!loc:@loss_1/dense_6_loss/Mean_1*
_output_shapes
:
И
Gtraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_1_grad/Fill/valueConst*
value	B :*
dtype0*-
_class#
!loc:@loss_1/dense_6_loss/Mean_1*
_output_shapes
: 
Њ
Atraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_1_grad/FillFillDtraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_1_grad/Shape_1Gtraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_1_grad/Fill/value*
T0*-
_class#
!loc:@loss_1/dense_6_loss/Mean_1*
_output_shapes
: 
Ю
Jtraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_1_grad/DynamicStitchDynamicStitchBtraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_1_grad/range@training_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_1_grad/modBtraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_1_grad/ShapeAtraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_1_grad/Fill*
N*
T0*-
_class#
!loc:@loss_1/dense_6_loss/Mean_1*#
_output_shapes
:џџџџџџџџџ
З
Ftraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_1_grad/Maximum/yConst*
value	B :*
dtype0*-
_class#
!loc:@loss_1/dense_6_loss/Mean_1*
_output_shapes
: 
Р
Dtraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_1_grad/MaximumMaximumJtraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_1_grad/DynamicStitchFtraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_1_grad/Maximum/y*
T0*-
_class#
!loc:@loss_1/dense_6_loss/Mean_1*#
_output_shapes
:џџџџџџџџџ
И
Etraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_1_grad/floordivFloorDivBtraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_1_grad/ShapeDtraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_1_grad/Maximum*
T0*-
_class#
!loc:@loss_1/dense_6_loss/Mean_1*#
_output_shapes
:џџџџџџџџџ
О
Dtraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_1_grad/ReshapeReshapeAtraining_1/RMSprop/gradients/loss_1/dense_6_loss/mul_grad/ReshapeJtraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_1_grad/DynamicStitch*
T0*
Tshape0*-
_class#
!loc:@loss_1/dense_6_loss/Mean_1*
_output_shapes
:
К
Atraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_1_grad/TileTileDtraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_1_grad/ReshapeEtraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_1_grad/floordiv*
T0*

Tmultiples0*-
_class#
!loc:@loss_1/dense_6_loss/Mean_1*
_output_shapes
:
Ы
Dtraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_1_grad/Shape_2Shapeloss_1/dense_6_loss/Mean*
T0*
out_type0*-
_class#
!loc:@loss_1/dense_6_loss/Mean_1*
_output_shapes
:
Э
Dtraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_1_grad/Shape_3Shapeloss_1/dense_6_loss/Mean_1*
T0*
out_type0*-
_class#
!loc:@loss_1/dense_6_loss/Mean_1*
_output_shapes
:
Л
Btraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_1_grad/ConstConst*
valueB: *
dtype0*-
_class#
!loc:@loss_1/dense_6_loss/Mean_1*
_output_shapes
:
Р
Atraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_1_grad/ProdProdDtraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_1_grad/Shape_2Btraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_1_grad/Const*
	keep_dims( *
T0*

Tidx0*-
_class#
!loc:@loss_1/dense_6_loss/Mean_1*
_output_shapes
: 
Н
Dtraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_1_grad/Const_1Const*
valueB: *
dtype0*-
_class#
!loc:@loss_1/dense_6_loss/Mean_1*
_output_shapes
:
Ф
Ctraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_1_grad/Prod_1ProdDtraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_1_grad/Shape_3Dtraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_1_grad/Const_1*
	keep_dims( *
T0*

Tidx0*-
_class#
!loc:@loss_1/dense_6_loss/Mean_1*
_output_shapes
: 
Й
Htraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_1_grad/Maximum_1/yConst*
value	B :*
dtype0*-
_class#
!loc:@loss_1/dense_6_loss/Mean_1*
_output_shapes
: 
А
Ftraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_1_grad/Maximum_1MaximumCtraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_1_grad/Prod_1Htraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_1_grad/Maximum_1/y*
T0*-
_class#
!loc:@loss_1/dense_6_loss/Mean_1*
_output_shapes
: 
Ў
Gtraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_1_grad/floordiv_1FloorDivAtraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_1_grad/ProdFtraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_1_grad/Maximum_1*
T0*-
_class#
!loc:@loss_1/dense_6_loss/Mean_1*
_output_shapes
: 
ё
Atraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_1_grad/CastCastGtraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_1_grad/floordiv_1*

SrcT0*

DstT0*-
_class#
!loc:@loss_1/dense_6_loss/Mean_1*
_output_shapes
: 
В
Dtraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_1_grad/truedivRealDivAtraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_1_grad/TileAtraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_1_grad/Cast*
T0*-
_class#
!loc:@loss_1/dense_6_loss/Mean_1*#
_output_shapes
:џџџџџџџџџ
Ч
@training_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_grad/ShapeShapeloss_1/dense_6_loss/Square*
T0*
out_type0*+
_class!
loc:@loss_1/dense_6_loss/Mean*
_output_shapes
:
Ў
?training_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_grad/SizeConst*
value	B :*
dtype0*+
_class!
loc:@loss_1/dense_6_loss/Mean*
_output_shapes
: 

>training_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_grad/addAdd*loss_1/dense_6_loss/Mean/reduction_indices?training_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_grad/Size*
T0*+
_class!
loc:@loss_1/dense_6_loss/Mean*
_output_shapes
: 

>training_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_grad/modFloorMod>training_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_grad/add?training_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_grad/Size*
T0*+
_class!
loc:@loss_1/dense_6_loss/Mean*
_output_shapes
: 
В
Btraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_grad/Shape_1Const*
valueB *
dtype0*+
_class!
loc:@loss_1/dense_6_loss/Mean*
_output_shapes
: 
Е
Ftraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_grad/range/startConst*
value	B : *
dtype0*+
_class!
loc:@loss_1/dense_6_loss/Mean*
_output_shapes
: 
Е
Ftraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_grad/range/deltaConst*
value	B :*
dtype0*+
_class!
loc:@loss_1/dense_6_loss/Mean*
_output_shapes
: 
я
@training_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_grad/rangeRangeFtraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_grad/range/start?training_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_grad/SizeFtraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_grad/range/delta*

Tidx0*+
_class!
loc:@loss_1/dense_6_loss/Mean*
_output_shapes
:
Д
Etraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_grad/Fill/valueConst*
value	B :*
dtype0*+
_class!
loc:@loss_1/dense_6_loss/Mean*
_output_shapes
: 
 
?training_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_grad/FillFillBtraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_grad/Shape_1Etraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_grad/Fill/value*
T0*+
_class!
loc:@loss_1/dense_6_loss/Mean*
_output_shapes
: 
Т
Htraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_grad/DynamicStitchDynamicStitch@training_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_grad/range>training_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_grad/mod@training_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_grad/Shape?training_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_grad/Fill*
N*
T0*+
_class!
loc:@loss_1/dense_6_loss/Mean*#
_output_shapes
:џџџџџџџџџ
Г
Dtraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_grad/Maximum/yConst*
value	B :*
dtype0*+
_class!
loc:@loss_1/dense_6_loss/Mean*
_output_shapes
: 
И
Btraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_grad/MaximumMaximumHtraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_grad/DynamicStitchDtraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_grad/Maximum/y*
T0*+
_class!
loc:@loss_1/dense_6_loss/Mean*#
_output_shapes
:џџџџџџџџџ
Ї
Ctraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_grad/floordivFloorDiv@training_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_grad/ShapeBtraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_grad/Maximum*
T0*+
_class!
loc:@loss_1/dense_6_loss/Mean*
_output_shapes
:
Л
Btraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_grad/ReshapeReshapeDtraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_1_grad/truedivHtraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_grad/DynamicStitch*
T0*
Tshape0*+
_class!
loc:@loss_1/dense_6_loss/Mean*
_output_shapes
:
Ъ
?training_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_grad/TileTileBtraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_grad/ReshapeCtraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_grad/floordiv*
T0*

Tmultiples0*+
_class!
loc:@loss_1/dense_6_loss/Mean*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
Щ
Btraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_grad/Shape_2Shapeloss_1/dense_6_loss/Square*
T0*
out_type0*+
_class!
loc:@loss_1/dense_6_loss/Mean*
_output_shapes
:
Ч
Btraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_grad/Shape_3Shapeloss_1/dense_6_loss/Mean*
T0*
out_type0*+
_class!
loc:@loss_1/dense_6_loss/Mean*
_output_shapes
:
З
@training_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_grad/ConstConst*
valueB: *
dtype0*+
_class!
loc:@loss_1/dense_6_loss/Mean*
_output_shapes
:
И
?training_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_grad/ProdProdBtraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_grad/Shape_2@training_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_grad/Const*
	keep_dims( *
T0*

Tidx0*+
_class!
loc:@loss_1/dense_6_loss/Mean*
_output_shapes
: 
Й
Btraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_grad/Const_1Const*
valueB: *
dtype0*+
_class!
loc:@loss_1/dense_6_loss/Mean*
_output_shapes
:
М
Atraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_grad/Prod_1ProdBtraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_grad/Shape_3Btraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_grad/Const_1*
	keep_dims( *
T0*

Tidx0*+
_class!
loc:@loss_1/dense_6_loss/Mean*
_output_shapes
: 
Е
Ftraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_grad/Maximum_1/yConst*
value	B :*
dtype0*+
_class!
loc:@loss_1/dense_6_loss/Mean*
_output_shapes
: 
Ј
Dtraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_grad/Maximum_1MaximumAtraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_grad/Prod_1Ftraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_grad/Maximum_1/y*
T0*+
_class!
loc:@loss_1/dense_6_loss/Mean*
_output_shapes
: 
І
Etraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_grad/floordiv_1FloorDiv?training_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_grad/ProdDtraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_grad/Maximum_1*
T0*+
_class!
loc:@loss_1/dense_6_loss/Mean*
_output_shapes
: 
ы
?training_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_grad/CastCastEtraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_grad/floordiv_1*

SrcT0*

DstT0*+
_class!
loc:@loss_1/dense_6_loss/Mean*
_output_shapes
: 
З
Btraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_grad/truedivRealDiv?training_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_grad/Tile?training_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_grad/Cast*
T0*+
_class!
loc:@loss_1/dense_6_loss/Mean*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
ћ
Btraining_1/RMSprop/gradients/loss_1/dense_6_loss/Square_grad/mul/xConstC^training_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_grad/truediv*
valueB
 *   @*
dtype0*-
_class#
!loc:@loss_1/dense_6_loss/Square*
_output_shapes
: 

@training_1/RMSprop/gradients/loss_1/dense_6_loss/Square_grad/mulMulBtraining_1/RMSprop/gradients/loss_1/dense_6_loss/Square_grad/mul/xloss_1/dense_6_loss/sub*
T0*-
_class#
!loc:@loss_1/dense_6_loss/Square*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
Й
Btraining_1/RMSprop/gradients/loss_1/dense_6_loss/Square_grad/mul_1MulBtraining_1/RMSprop/gradients/loss_1/dense_6_loss/Mean_grad/truediv@training_1/RMSprop/gradients/loss_1/dense_6_loss/Square_grad/mul*
T0*-
_class#
!loc:@loss_1/dense_6_loss/Square*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
К
?training_1/RMSprop/gradients/loss_1/dense_6_loss/sub_grad/ShapeShapedense_6/BiasAdd*
T0*
out_type0**
_class 
loc:@loss_1/dense_6_loss/sub*
_output_shapes
:
Л
Atraining_1/RMSprop/gradients/loss_1/dense_6_loss/sub_grad/Shape_1Shapedense_6_target*
T0*
out_type0**
_class 
loc:@loss_1/dense_6_loss/sub*
_output_shapes
:
е
Otraining_1/RMSprop/gradients/loss_1/dense_6_loss/sub_grad/BroadcastGradientArgsBroadcastGradientArgs?training_1/RMSprop/gradients/loss_1/dense_6_loss/sub_grad/ShapeAtraining_1/RMSprop/gradients/loss_1/dense_6_loss/sub_grad/Shape_1*
T0**
_class 
loc:@loss_1/dense_6_loss/sub*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Х
=training_1/RMSprop/gradients/loss_1/dense_6_loss/sub_grad/SumSumBtraining_1/RMSprop/gradients/loss_1/dense_6_loss/Square_grad/mul_1Otraining_1/RMSprop/gradients/loss_1/dense_6_loss/sub_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0**
_class 
loc:@loss_1/dense_6_loss/sub*
_output_shapes
:
И
Atraining_1/RMSprop/gradients/loss_1/dense_6_loss/sub_grad/ReshapeReshape=training_1/RMSprop/gradients/loss_1/dense_6_loss/sub_grad/Sum?training_1/RMSprop/gradients/loss_1/dense_6_loss/sub_grad/Shape*
T0*
Tshape0**
_class 
loc:@loss_1/dense_6_loss/sub*'
_output_shapes
:џџџџџџџџџ
Щ
?training_1/RMSprop/gradients/loss_1/dense_6_loss/sub_grad/Sum_1SumBtraining_1/RMSprop/gradients/loss_1/dense_6_loss/Square_grad/mul_1Qtraining_1/RMSprop/gradients/loss_1/dense_6_loss/sub_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0**
_class 
loc:@loss_1/dense_6_loss/sub*
_output_shapes
:
д
=training_1/RMSprop/gradients/loss_1/dense_6_loss/sub_grad/NegNeg?training_1/RMSprop/gradients/loss_1/dense_6_loss/sub_grad/Sum_1*
T0**
_class 
loc:@loss_1/dense_6_loss/sub*
_output_shapes
:
Х
Ctraining_1/RMSprop/gradients/loss_1/dense_6_loss/sub_grad/Reshape_1Reshape=training_1/RMSprop/gradients/loss_1/dense_6_loss/sub_grad/NegAtraining_1/RMSprop/gradients/loss_1/dense_6_loss/sub_grad/Shape_1*
T0*
Tshape0**
_class 
loc:@loss_1/dense_6_loss/sub*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
я
=training_1/RMSprop/gradients/dense_6/BiasAdd_grad/BiasAddGradBiasAddGradAtraining_1/RMSprop/gradients/loss_1/dense_6_loss/sub_grad/Reshape*
T0*
data_formatNHWC*"
_class
loc:@dense_6/BiasAdd*
_output_shapes
:

7training_1/RMSprop/gradients/dense_6/MatMul_grad/MatMulMatMulAtraining_1/RMSprop/gradients/loss_1/dense_6_loss/sub_grad/Reshapedense_6/kernel/read*
transpose_a( *
transpose_b(*
T0*!
_class
loc:@dense_6/MatMul*'
_output_shapes
:џџџџџџџџџ@

9training_1/RMSprop/gradients/dense_6/MatMul_grad/MatMul_1MatMuldense_5/ReluAtraining_1/RMSprop/gradients/loss_1/dense_6_loss/sub_grad/Reshape*
transpose_a(*
transpose_b( *
T0*!
_class
loc:@dense_6/MatMul*
_output_shapes

:@
н
7training_1/RMSprop/gradients/dense_5/Relu_grad/ReluGradReluGrad7training_1/RMSprop/gradients/dense_6/MatMul_grad/MatMuldense_5/Relu*
T0*
_class
loc:@dense_5/Relu*'
_output_shapes
:џџџџџџџџџ@
х
=training_1/RMSprop/gradients/dense_5/BiasAdd_grad/BiasAddGradBiasAddGrad7training_1/RMSprop/gradients/dense_5/Relu_grad/ReluGrad*
T0*
data_formatNHWC*"
_class
loc:@dense_5/BiasAdd*
_output_shapes
:@

7training_1/RMSprop/gradients/dense_5/MatMul_grad/MatMulMatMul7training_1/RMSprop/gradients/dense_5/Relu_grad/ReluGraddense_5/kernel/read*
transpose_a( *
transpose_b(*
T0*!
_class
loc:@dense_5/MatMul*'
_output_shapes
:џџџџџџџџџ@
ќ
9training_1/RMSprop/gradients/dense_5/MatMul_grad/MatMul_1MatMuldense_4/Relu7training_1/RMSprop/gradients/dense_5/Relu_grad/ReluGrad*
transpose_a(*
transpose_b( *
T0*!
_class
loc:@dense_5/MatMul*
_output_shapes

:@@
н
7training_1/RMSprop/gradients/dense_4/Relu_grad/ReluGradReluGrad7training_1/RMSprop/gradients/dense_5/MatMul_grad/MatMuldense_4/Relu*
T0*
_class
loc:@dense_4/Relu*'
_output_shapes
:џџџџџџџџџ@
х
=training_1/RMSprop/gradients/dense_4/BiasAdd_grad/BiasAddGradBiasAddGrad7training_1/RMSprop/gradients/dense_4/Relu_grad/ReluGrad*
T0*
data_formatNHWC*"
_class
loc:@dense_4/BiasAdd*
_output_shapes
:@

7training_1/RMSprop/gradients/dense_4/MatMul_grad/MatMulMatMul7training_1/RMSprop/gradients/dense_4/Relu_grad/ReluGraddense_4/kernel/read*
transpose_a( *
transpose_b(*
T0*!
_class
loc:@dense_4/MatMul*'
_output_shapes
:џџџџџџџџџ
§
9training_1/RMSprop/gradients/dense_4/MatMul_grad/MatMul_1MatMuldense_4_input7training_1/RMSprop/gradients/dense_4/Relu_grad/ReluGrad*
transpose_a(*
transpose_b( *
T0*!
_class
loc:@dense_4/MatMul*
_output_shapes

:@
m
training_1/RMSprop/ConstConst*
valueB@*    *
dtype0*
_output_shapes

:@

training_1/RMSprop/Variable
VariableV2*
shape
:@*
dtype0*
	container *
shared_name *
_output_shapes

:@
х
"training_1/RMSprop/Variable/AssignAssigntraining_1/RMSprop/Variabletraining_1/RMSprop/Const*
T0*
validate_shape(*
use_locking(*.
_class$
" loc:@training_1/RMSprop/Variable*
_output_shapes

:@
Ђ
 training_1/RMSprop/Variable/readIdentitytraining_1/RMSprop/Variable*
T0*.
_class$
" loc:@training_1/RMSprop/Variable*
_output_shapes

:@
g
training_1/RMSprop/Const_1Const*
valueB@*    *
dtype0*
_output_shapes
:@

training_1/RMSprop/Variable_1
VariableV2*
shape:@*
dtype0*
	container *
shared_name *
_output_shapes
:@
щ
$training_1/RMSprop/Variable_1/AssignAssigntraining_1/RMSprop/Variable_1training_1/RMSprop/Const_1*
T0*
validate_shape(*
use_locking(*0
_class&
$"loc:@training_1/RMSprop/Variable_1*
_output_shapes
:@
Є
"training_1/RMSprop/Variable_1/readIdentitytraining_1/RMSprop/Variable_1*
T0*0
_class&
$"loc:@training_1/RMSprop/Variable_1*
_output_shapes
:@
o
training_1/RMSprop/Const_2Const*
valueB@@*    *
dtype0*
_output_shapes

:@@

training_1/RMSprop/Variable_2
VariableV2*
shape
:@@*
dtype0*
	container *
shared_name *
_output_shapes

:@@
э
$training_1/RMSprop/Variable_2/AssignAssigntraining_1/RMSprop/Variable_2training_1/RMSprop/Const_2*
T0*
validate_shape(*
use_locking(*0
_class&
$"loc:@training_1/RMSprop/Variable_2*
_output_shapes

:@@
Ј
"training_1/RMSprop/Variable_2/readIdentitytraining_1/RMSprop/Variable_2*
T0*0
_class&
$"loc:@training_1/RMSprop/Variable_2*
_output_shapes

:@@
g
training_1/RMSprop/Const_3Const*
valueB@*    *
dtype0*
_output_shapes
:@

training_1/RMSprop/Variable_3
VariableV2*
shape:@*
dtype0*
	container *
shared_name *
_output_shapes
:@
щ
$training_1/RMSprop/Variable_3/AssignAssigntraining_1/RMSprop/Variable_3training_1/RMSprop/Const_3*
T0*
validate_shape(*
use_locking(*0
_class&
$"loc:@training_1/RMSprop/Variable_3*
_output_shapes
:@
Є
"training_1/RMSprop/Variable_3/readIdentitytraining_1/RMSprop/Variable_3*
T0*0
_class&
$"loc:@training_1/RMSprop/Variable_3*
_output_shapes
:@
o
training_1/RMSprop/Const_4Const*
valueB@*    *
dtype0*
_output_shapes

:@

training_1/RMSprop/Variable_4
VariableV2*
shape
:@*
dtype0*
	container *
shared_name *
_output_shapes

:@
э
$training_1/RMSprop/Variable_4/AssignAssigntraining_1/RMSprop/Variable_4training_1/RMSprop/Const_4*
T0*
validate_shape(*
use_locking(*0
_class&
$"loc:@training_1/RMSprop/Variable_4*
_output_shapes

:@
Ј
"training_1/RMSprop/Variable_4/readIdentitytraining_1/RMSprop/Variable_4*
T0*0
_class&
$"loc:@training_1/RMSprop/Variable_4*
_output_shapes

:@
g
training_1/RMSprop/Const_5Const*
valueB*    *
dtype0*
_output_shapes
:

training_1/RMSprop/Variable_5
VariableV2*
shape:*
dtype0*
	container *
shared_name *
_output_shapes
:
щ
$training_1/RMSprop/Variable_5/AssignAssigntraining_1/RMSprop/Variable_5training_1/RMSprop/Const_5*
T0*
validate_shape(*
use_locking(*0
_class&
$"loc:@training_1/RMSprop/Variable_5*
_output_shapes
:
Є
"training_1/RMSprop/Variable_5/readIdentitytraining_1/RMSprop/Variable_5*
T0*0
_class&
$"loc:@training_1/RMSprop/Variable_5*
_output_shapes
:
d
"training_1/RMSprop/AssignAdd/valueConst*
value	B	 R*
dtype0	*
_output_shapes
: 
Р
training_1/RMSprop/AssignAdd	AssignAddRMSprop_1/iterations"training_1/RMSprop/AssignAdd/value*
T0	*
use_locking( *'
_class
loc:@RMSprop_1/iterations*
_output_shapes
: 
|
training_1/RMSprop/mulMulRMSprop_1/rho/read training_1/RMSprop/Variable/read*
T0*
_output_shapes

:@
]
training_1/RMSprop/sub/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
l
training_1/RMSprop/subSubtraining_1/RMSprop/sub/xRMSprop_1/rho/read*
T0*
_output_shapes
: 

training_1/RMSprop/SquareSquare9training_1/RMSprop/gradients/dense_4/MatMul_grad/MatMul_1*
T0*
_output_shapes

:@
{
training_1/RMSprop/mul_1Multraining_1/RMSprop/subtraining_1/RMSprop/Square*
T0*
_output_shapes

:@
x
training_1/RMSprop/addAddtraining_1/RMSprop/multraining_1/RMSprop/mul_1*
T0*
_output_shapes

:@
к
training_1/RMSprop/AssignAssigntraining_1/RMSprop/Variabletraining_1/RMSprop/add*
T0*
validate_shape(*
use_locking(*.
_class$
" loc:@training_1/RMSprop/Variable*
_output_shapes

:@

training_1/RMSprop/mul_2MulRMSprop_1/lr/read9training_1/RMSprop/gradients/dense_4/MatMul_grad/MatMul_1*
T0*
_output_shapes

:@
_
training_1/RMSprop/Const_6Const*
valueB
 *    *
dtype0*
_output_shapes
: 
_
training_1/RMSprop/Const_7Const*
valueB
 *  *
dtype0*
_output_shapes
: 

(training_1/RMSprop/clip_by_value/MinimumMinimumtraining_1/RMSprop/addtraining_1/RMSprop/Const_7*
T0*
_output_shapes

:@

 training_1/RMSprop/clip_by_valueMaximum(training_1/RMSprop/clip_by_value/Minimumtraining_1/RMSprop/Const_6*
T0*
_output_shapes

:@
j
training_1/RMSprop/SqrtSqrt training_1/RMSprop/clip_by_value*
T0*
_output_shapes

:@
_
training_1/RMSprop/add_1/yConst*
valueB
 *wЬ+2*
dtype0*
_output_shapes
: 
}
training_1/RMSprop/add_1Addtraining_1/RMSprop/Sqrttraining_1/RMSprop/add_1/y*
T0*
_output_shapes

:@

training_1/RMSprop/truedivRealDivtraining_1/RMSprop/mul_2training_1/RMSprop/add_1*
T0*
_output_shapes

:@
y
training_1/RMSprop/sub_1Subdense_4/kernel/readtraining_1/RMSprop/truediv*
T0*
_output_shapes

:@
Ф
training_1/RMSprop/Assign_1Assigndense_4/kerneltraining_1/RMSprop/sub_1*
T0*
validate_shape(*
use_locking(*!
_class
loc:@dense_4/kernel*
_output_shapes

:@
|
training_1/RMSprop/mul_3MulRMSprop_1/rho/read"training_1/RMSprop/Variable_1/read*
T0*
_output_shapes
:@
_
training_1/RMSprop/sub_2/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
p
training_1/RMSprop/sub_2Subtraining_1/RMSprop/sub_2/xRMSprop_1/rho/read*
T0*
_output_shapes
: 

training_1/RMSprop/Square_1Square=training_1/RMSprop/gradients/dense_4/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:@
{
training_1/RMSprop/mul_4Multraining_1/RMSprop/sub_2training_1/RMSprop/Square_1*
T0*
_output_shapes
:@
x
training_1/RMSprop/add_2Addtraining_1/RMSprop/mul_3training_1/RMSprop/mul_4*
T0*
_output_shapes
:@
о
training_1/RMSprop/Assign_2Assigntraining_1/RMSprop/Variable_1training_1/RMSprop/add_2*
T0*
validate_shape(*
use_locking(*0
_class&
$"loc:@training_1/RMSprop/Variable_1*
_output_shapes
:@

training_1/RMSprop/mul_5MulRMSprop_1/lr/read=training_1/RMSprop/gradients/dense_4/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:@
_
training_1/RMSprop/Const_8Const*
valueB
 *    *
dtype0*
_output_shapes
: 
_
training_1/RMSprop/Const_9Const*
valueB
 *  *
dtype0*
_output_shapes
: 

*training_1/RMSprop/clip_by_value_1/MinimumMinimumtraining_1/RMSprop/add_2training_1/RMSprop/Const_9*
T0*
_output_shapes
:@

"training_1/RMSprop/clip_by_value_1Maximum*training_1/RMSprop/clip_by_value_1/Minimumtraining_1/RMSprop/Const_8*
T0*
_output_shapes
:@
j
training_1/RMSprop/Sqrt_1Sqrt"training_1/RMSprop/clip_by_value_1*
T0*
_output_shapes
:@
_
training_1/RMSprop/add_3/yConst*
valueB
 *wЬ+2*
dtype0*
_output_shapes
: 
{
training_1/RMSprop/add_3Addtraining_1/RMSprop/Sqrt_1training_1/RMSprop/add_3/y*
T0*
_output_shapes
:@

training_1/RMSprop/truediv_1RealDivtraining_1/RMSprop/mul_5training_1/RMSprop/add_3*
T0*
_output_shapes
:@
u
training_1/RMSprop/sub_3Subdense_4/bias/readtraining_1/RMSprop/truediv_1*
T0*
_output_shapes
:@
М
training_1/RMSprop/Assign_3Assigndense_4/biastraining_1/RMSprop/sub_3*
T0*
validate_shape(*
use_locking(*
_class
loc:@dense_4/bias*
_output_shapes
:@

training_1/RMSprop/mul_6MulRMSprop_1/rho/read"training_1/RMSprop/Variable_2/read*
T0*
_output_shapes

:@@
_
training_1/RMSprop/sub_4/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
p
training_1/RMSprop/sub_4Subtraining_1/RMSprop/sub_4/xRMSprop_1/rho/read*
T0*
_output_shapes
: 

training_1/RMSprop/Square_2Square9training_1/RMSprop/gradients/dense_5/MatMul_grad/MatMul_1*
T0*
_output_shapes

:@@

training_1/RMSprop/mul_7Multraining_1/RMSprop/sub_4training_1/RMSprop/Square_2*
T0*
_output_shapes

:@@
|
training_1/RMSprop/add_4Addtraining_1/RMSprop/mul_6training_1/RMSprop/mul_7*
T0*
_output_shapes

:@@
т
training_1/RMSprop/Assign_4Assigntraining_1/RMSprop/Variable_2training_1/RMSprop/add_4*
T0*
validate_shape(*
use_locking(*0
_class&
$"loc:@training_1/RMSprop/Variable_2*
_output_shapes

:@@

training_1/RMSprop/mul_8MulRMSprop_1/lr/read9training_1/RMSprop/gradients/dense_5/MatMul_grad/MatMul_1*
T0*
_output_shapes

:@@
`
training_1/RMSprop/Const_10Const*
valueB
 *    *
dtype0*
_output_shapes
: 
`
training_1/RMSprop/Const_11Const*
valueB
 *  *
dtype0*
_output_shapes
: 

*training_1/RMSprop/clip_by_value_2/MinimumMinimumtraining_1/RMSprop/add_4training_1/RMSprop/Const_11*
T0*
_output_shapes

:@@

"training_1/RMSprop/clip_by_value_2Maximum*training_1/RMSprop/clip_by_value_2/Minimumtraining_1/RMSprop/Const_10*
T0*
_output_shapes

:@@
n
training_1/RMSprop/Sqrt_2Sqrt"training_1/RMSprop/clip_by_value_2*
T0*
_output_shapes

:@@
_
training_1/RMSprop/add_5/yConst*
valueB
 *wЬ+2*
dtype0*
_output_shapes
: 

training_1/RMSprop/add_5Addtraining_1/RMSprop/Sqrt_2training_1/RMSprop/add_5/y*
T0*
_output_shapes

:@@

training_1/RMSprop/truediv_2RealDivtraining_1/RMSprop/mul_8training_1/RMSprop/add_5*
T0*
_output_shapes

:@@
{
training_1/RMSprop/sub_5Subdense_5/kernel/readtraining_1/RMSprop/truediv_2*
T0*
_output_shapes

:@@
Ф
training_1/RMSprop/Assign_5Assigndense_5/kerneltraining_1/RMSprop/sub_5*
T0*
validate_shape(*
use_locking(*!
_class
loc:@dense_5/kernel*
_output_shapes

:@@
|
training_1/RMSprop/mul_9MulRMSprop_1/rho/read"training_1/RMSprop/Variable_3/read*
T0*
_output_shapes
:@
_
training_1/RMSprop/sub_6/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
p
training_1/RMSprop/sub_6Subtraining_1/RMSprop/sub_6/xRMSprop_1/rho/read*
T0*
_output_shapes
: 

training_1/RMSprop/Square_3Square=training_1/RMSprop/gradients/dense_5/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:@
|
training_1/RMSprop/mul_10Multraining_1/RMSprop/sub_6training_1/RMSprop/Square_3*
T0*
_output_shapes
:@
y
training_1/RMSprop/add_6Addtraining_1/RMSprop/mul_9training_1/RMSprop/mul_10*
T0*
_output_shapes
:@
о
training_1/RMSprop/Assign_6Assigntraining_1/RMSprop/Variable_3training_1/RMSprop/add_6*
T0*
validate_shape(*
use_locking(*0
_class&
$"loc:@training_1/RMSprop/Variable_3*
_output_shapes
:@

training_1/RMSprop/mul_11MulRMSprop_1/lr/read=training_1/RMSprop/gradients/dense_5/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:@
`
training_1/RMSprop/Const_12Const*
valueB
 *    *
dtype0*
_output_shapes
: 
`
training_1/RMSprop/Const_13Const*
valueB
 *  *
dtype0*
_output_shapes
: 

*training_1/RMSprop/clip_by_value_3/MinimumMinimumtraining_1/RMSprop/add_6training_1/RMSprop/Const_13*
T0*
_output_shapes
:@

"training_1/RMSprop/clip_by_value_3Maximum*training_1/RMSprop/clip_by_value_3/Minimumtraining_1/RMSprop/Const_12*
T0*
_output_shapes
:@
j
training_1/RMSprop/Sqrt_3Sqrt"training_1/RMSprop/clip_by_value_3*
T0*
_output_shapes
:@
_
training_1/RMSprop/add_7/yConst*
valueB
 *wЬ+2*
dtype0*
_output_shapes
: 
{
training_1/RMSprop/add_7Addtraining_1/RMSprop/Sqrt_3training_1/RMSprop/add_7/y*
T0*
_output_shapes
:@

training_1/RMSprop/truediv_3RealDivtraining_1/RMSprop/mul_11training_1/RMSprop/add_7*
T0*
_output_shapes
:@
u
training_1/RMSprop/sub_7Subdense_5/bias/readtraining_1/RMSprop/truediv_3*
T0*
_output_shapes
:@
М
training_1/RMSprop/Assign_7Assigndense_5/biastraining_1/RMSprop/sub_7*
T0*
validate_shape(*
use_locking(*
_class
loc:@dense_5/bias*
_output_shapes
:@

training_1/RMSprop/mul_12MulRMSprop_1/rho/read"training_1/RMSprop/Variable_4/read*
T0*
_output_shapes

:@
_
training_1/RMSprop/sub_8/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
p
training_1/RMSprop/sub_8Subtraining_1/RMSprop/sub_8/xRMSprop_1/rho/read*
T0*
_output_shapes
: 

training_1/RMSprop/Square_4Square9training_1/RMSprop/gradients/dense_6/MatMul_grad/MatMul_1*
T0*
_output_shapes

:@

training_1/RMSprop/mul_13Multraining_1/RMSprop/sub_8training_1/RMSprop/Square_4*
T0*
_output_shapes

:@
~
training_1/RMSprop/add_8Addtraining_1/RMSprop/mul_12training_1/RMSprop/mul_13*
T0*
_output_shapes

:@
т
training_1/RMSprop/Assign_8Assigntraining_1/RMSprop/Variable_4training_1/RMSprop/add_8*
T0*
validate_shape(*
use_locking(*0
_class&
$"loc:@training_1/RMSprop/Variable_4*
_output_shapes

:@

training_1/RMSprop/mul_14MulRMSprop_1/lr/read9training_1/RMSprop/gradients/dense_6/MatMul_grad/MatMul_1*
T0*
_output_shapes

:@
`
training_1/RMSprop/Const_14Const*
valueB
 *    *
dtype0*
_output_shapes
: 
`
training_1/RMSprop/Const_15Const*
valueB
 *  *
dtype0*
_output_shapes
: 

*training_1/RMSprop/clip_by_value_4/MinimumMinimumtraining_1/RMSprop/add_8training_1/RMSprop/Const_15*
T0*
_output_shapes

:@

"training_1/RMSprop/clip_by_value_4Maximum*training_1/RMSprop/clip_by_value_4/Minimumtraining_1/RMSprop/Const_14*
T0*
_output_shapes

:@
n
training_1/RMSprop/Sqrt_4Sqrt"training_1/RMSprop/clip_by_value_4*
T0*
_output_shapes

:@
_
training_1/RMSprop/add_9/yConst*
valueB
 *wЬ+2*
dtype0*
_output_shapes
: 

training_1/RMSprop/add_9Addtraining_1/RMSprop/Sqrt_4training_1/RMSprop/add_9/y*
T0*
_output_shapes

:@

training_1/RMSprop/truediv_4RealDivtraining_1/RMSprop/mul_14training_1/RMSprop/add_9*
T0*
_output_shapes

:@
{
training_1/RMSprop/sub_9Subdense_6/kernel/readtraining_1/RMSprop/truediv_4*
T0*
_output_shapes

:@
Ф
training_1/RMSprop/Assign_9Assigndense_6/kerneltraining_1/RMSprop/sub_9*
T0*
validate_shape(*
use_locking(*!
_class
loc:@dense_6/kernel*
_output_shapes

:@
}
training_1/RMSprop/mul_15MulRMSprop_1/rho/read"training_1/RMSprop/Variable_5/read*
T0*
_output_shapes
:
`
training_1/RMSprop/sub_10/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
r
training_1/RMSprop/sub_10Subtraining_1/RMSprop/sub_10/xRMSprop_1/rho/read*
T0*
_output_shapes
: 

training_1/RMSprop/Square_5Square=training_1/RMSprop/gradients/dense_6/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:
}
training_1/RMSprop/mul_16Multraining_1/RMSprop/sub_10training_1/RMSprop/Square_5*
T0*
_output_shapes
:
{
training_1/RMSprop/add_10Addtraining_1/RMSprop/mul_15training_1/RMSprop/mul_16*
T0*
_output_shapes
:
р
training_1/RMSprop/Assign_10Assigntraining_1/RMSprop/Variable_5training_1/RMSprop/add_10*
T0*
validate_shape(*
use_locking(*0
_class&
$"loc:@training_1/RMSprop/Variable_5*
_output_shapes
:

training_1/RMSprop/mul_17MulRMSprop_1/lr/read=training_1/RMSprop/gradients/dense_6/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:
`
training_1/RMSprop/Const_16Const*
valueB
 *    *
dtype0*
_output_shapes
: 
`
training_1/RMSprop/Const_17Const*
valueB
 *  *
dtype0*
_output_shapes
: 

*training_1/RMSprop/clip_by_value_5/MinimumMinimumtraining_1/RMSprop/add_10training_1/RMSprop/Const_17*
T0*
_output_shapes
:

"training_1/RMSprop/clip_by_value_5Maximum*training_1/RMSprop/clip_by_value_5/Minimumtraining_1/RMSprop/Const_16*
T0*
_output_shapes
:
j
training_1/RMSprop/Sqrt_5Sqrt"training_1/RMSprop/clip_by_value_5*
T0*
_output_shapes
:
`
training_1/RMSprop/add_11/yConst*
valueB
 *wЬ+2*
dtype0*
_output_shapes
: 
}
training_1/RMSprop/add_11Addtraining_1/RMSprop/Sqrt_5training_1/RMSprop/add_11/y*
T0*
_output_shapes
:

training_1/RMSprop/truediv_5RealDivtraining_1/RMSprop/mul_17training_1/RMSprop/add_11*
T0*
_output_shapes
:
v
training_1/RMSprop/sub_11Subdense_6/bias/readtraining_1/RMSprop/truediv_5*
T0*
_output_shapes
:
О
training_1/RMSprop/Assign_11Assigndense_6/biastraining_1/RMSprop/sub_11*
T0*
validate_shape(*
use_locking(*
_class
loc:@dense_6/bias*
_output_shapes
:
и
training_1/group_depsNoOp^loss_1/mul%^metrics_1/mean_absolute_error/Mean_1^training_1/RMSprop/AssignAdd^training_1/RMSprop/Assign^training_1/RMSprop/Assign_1^training_1/RMSprop/Assign_2^training_1/RMSprop/Assign_3^training_1/RMSprop/Assign_4^training_1/RMSprop/Assign_5^training_1/RMSprop/Assign_6^training_1/RMSprop/Assign_7^training_1/RMSprop/Assign_8^training_1/RMSprop/Assign_9^training_1/RMSprop/Assign_10^training_1/RMSprop/Assign_11
H
group_deps_1NoOp^loss_1/mul%^metrics_1/mean_absolute_error/Mean_1

IsVariableInitialized_16IsVariableInitializeddense_4/kernel*
dtype0*!
_class
loc:@dense_4/kernel*
_output_shapes
: 

IsVariableInitialized_17IsVariableInitializeddense_4/bias*
dtype0*
_class
loc:@dense_4/bias*
_output_shapes
: 

IsVariableInitialized_18IsVariableInitializeddense_5/kernel*
dtype0*!
_class
loc:@dense_5/kernel*
_output_shapes
: 

IsVariableInitialized_19IsVariableInitializeddense_5/bias*
dtype0*
_class
loc:@dense_5/bias*
_output_shapes
: 

IsVariableInitialized_20IsVariableInitializeddense_6/kernel*
dtype0*!
_class
loc:@dense_6/kernel*
_output_shapes
: 

IsVariableInitialized_21IsVariableInitializeddense_6/bias*
dtype0*
_class
loc:@dense_6/bias*
_output_shapes
: 

IsVariableInitialized_22IsVariableInitializedRMSprop_1/lr*
dtype0*
_class
loc:@RMSprop_1/lr*
_output_shapes
: 

IsVariableInitialized_23IsVariableInitializedRMSprop_1/rho*
dtype0* 
_class
loc:@RMSprop_1/rho*
_output_shapes
: 

IsVariableInitialized_24IsVariableInitializedRMSprop_1/decay*
dtype0*"
_class
loc:@RMSprop_1/decay*
_output_shapes
: 

IsVariableInitialized_25IsVariableInitializedRMSprop_1/iterations*
dtype0	*'
_class
loc:@RMSprop_1/iterations*
_output_shapes
: 
Ѓ
IsVariableInitialized_26IsVariableInitializedtraining_1/RMSprop/Variable*
dt