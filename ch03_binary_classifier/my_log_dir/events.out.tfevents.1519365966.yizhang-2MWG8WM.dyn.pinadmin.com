       £K"	  А”л£÷Abrain.Event:2ъСCГ6ь      7—©	vЄ”л£÷A"©ш
p
dense_1_inputPlaceholder*
dtype0*
shape:€€€€€€€€€*'
_output_shapes
:€€€€€€€€€
m
dense_1/random_uniform/shapeConst*
valueB"   @   *
dtype0*
_output_shapes
:
_
dense_1/random_uniform/minConst*
valueB
 *!мОЊ*
dtype0*
_output_shapes
: 
_
dense_1/random_uniform/maxConst*
valueB
 *!мО>*
dtype0*
_output_shapes
: 
®
$dense_1/random_uniform/RandomUniformRandomUniformdense_1/random_uniform/shape*
seed±€е)*
seed2ічЩ*
dtype0*
T0*
_output_shapes

:@
z
dense_1/random_uniform/subSubdense_1/random_uniform/maxdense_1/random_uniform/min*
T0*
_output_shapes
: 
М
dense_1/random_uniform/mulMul$dense_1/random_uniform/RandomUniformdense_1/random_uniform/sub*
T0*
_output_shapes

:@
~
dense_1/random_uniformAdddense_1/random_uniform/muldense_1/random_uniform/min*
T0*
_output_shapes

:@
В
dense_1/kernel
VariableV2*
shape
:@*
dtype0*
	container *
shared_name *
_output_shapes

:@
Љ
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
©
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
Ф
dense_1/MatMulMatMuldense_1_inputdense_1/kernel/read*
transpose_a( *
transpose_b( *
T0*'
_output_shapes
:€€€€€€€€€@
Ж
dense_1/BiasAddBiasAdddense_1/MatMuldense_1/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:€€€€€€€€€@
W
dense_1/ReluReludense_1/BiasAdd*
T0*'
_output_shapes
:€€€€€€€€€@
m
dense_2/random_uniform/shapeConst*
valueB"@   @   *
dtype0*
_output_shapes
:
_
dense_2/random_uniform/minConst*
valueB
 *„≥]Њ*
dtype0*
_output_shapes
: 
_
dense_2/random_uniform/maxConst*
valueB
 *„≥]>*
dtype0*
_output_shapes
: 
®
$dense_2/random_uniform/RandomUniformRandomUniformdense_2/random_uniform/shape*
seed±€е)*
seed2€єх*
dtype0*
T0*
_output_shapes

:@@
z
dense_2/random_uniform/subSubdense_2/random_uniform/maxdense_2/random_uniform/min*
T0*
_output_shapes
: 
М
dense_2/random_uniform/mulMul$dense_2/random_uniform/RandomUniformdense_2/random_uniform/sub*
T0*
_output_shapes

:@@
~
dense_2/random_uniformAdddense_2/random_uniform/muldense_2/random_uniform/min*
T0*
_output_shapes

:@@
В
dense_2/kernel
VariableV2*
shape
:@@*
dtype0*
	container *
shared_name *
_output_shapes

:@@
Љ
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
©
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
У
dense_2/MatMulMatMuldense_1/Reludense_2/kernel/read*
transpose_a( *
transpose_b( *
T0*'
_output_shapes
:€€€€€€€€€@
Ж
dense_2/BiasAddBiasAdddense_2/MatMuldense_2/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:€€€€€€€€€@
W
dense_2/ReluReludense_2/BiasAdd*
T0*'
_output_shapes
:€€€€€€€€€@
m
dense_3/random_uniform/shapeConst*
valueB"@      *
dtype0*
_output_shapes
:
_
dense_3/random_uniform/minConst*
valueB
 *ИОЫЊ*
dtype0*
_output_shapes
: 
_
dense_3/random_uniform/maxConst*
valueB
 *ИОЫ>*
dtype0*
_output_shapes
: 
®
$dense_3/random_uniform/RandomUniformRandomUniformdense_3/random_uniform/shape*
seed±€е)*
seed2ицЯ*
dtype0*
T0*
_output_shapes

:@
z
dense_3/random_uniform/subSubdense_3/random_uniform/maxdense_3/random_uniform/min*
T0*
_output_shapes
: 
М
dense_3/random_uniform/mulMul$dense_3/random_uniform/RandomUniformdense_3/random_uniform/sub*
T0*
_output_shapes

:@
~
dense_3/random_uniformAdddense_3/random_uniform/muldense_3/random_uniform/min*
T0*
_output_shapes

:@
В
dense_3/kernel
VariableV2*
shape
:@*
dtype0*
	container *
shared_name *
_output_shapes

:@
Љ
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
©
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
У
dense_3/MatMulMatMuldense_2/Reludense_3/kernel/read*
transpose_a( *
transpose_b( *
T0*'
_output_shapes
:€€€€€€€€€
Ж
dense_3/BiasAddBiasAdddense_3/MatMuldense_3/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:€€€€€€€€€
]
RMSprop/lr/initial_valueConst*
valueB
 *oГ:*
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
™
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
Ѓ
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
ґ
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
 
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
Г
dense_3_targetPlaceholder*
dtype0*%
shape:€€€€€€€€€€€€€€€€€€*0
_output_shapes
:€€€€€€€€€€€€€€€€€€
q
dense_3_sample_weightsPlaceholder*
dtype0*
shape:€€€€€€€€€*#
_output_shapes
:€€€€€€€€€
x
loss/dense_3_loss/subSubdense_3/BiasAdddense_3_target*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€
t
loss/dense_3_loss/SquareSquareloss/dense_3_loss/sub*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€
s
(loss/dense_3_loss/Mean/reduction_indicesConst*
valueB :
€€€€€€€€€*
dtype0*
_output_shapes
: 
≠
loss/dense_3_loss/MeanMeanloss/dense_3_loss/Square(loss/dense_3_loss/Mean/reduction_indices*
	keep_dims( *
T0*

Tidx0*#
_output_shapes
:€€€€€€€€€
m
*loss/dense_3_loss/Mean_1/reduction_indicesConst*
valueB *
dtype0*
_output_shapes
: 
ѓ
loss/dense_3_loss/Mean_1Meanloss/dense_3_loss/Mean*loss/dense_3_loss/Mean_1/reduction_indices*
	keep_dims( *
T0*

Tidx0*#
_output_shapes
:€€€€€€€€€
|
loss/dense_3_loss/mulMulloss/dense_3_loss/Mean_1dense_3_sample_weights*
T0*#
_output_shapes
:€€€€€€€€€
a
loss/dense_3_loss/NotEqual/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
К
loss/dense_3_loss/NotEqualNotEqualdense_3_sample_weightsloss/dense_3_loss/NotEqual/y*
T0*#
_output_shapes
:€€€€€€€€€
w
loss/dense_3_loss/CastCastloss/dense_3_loss/NotEqual*

SrcT0
*

DstT0*#
_output_shapes
:€€€€€€€€€
a
loss/dense_3_loss/ConstConst*
valueB: *
dtype0*
_output_shapes
:
П
loss/dense_3_loss/Mean_2Meanloss/dense_3_loss/Castloss/dense_3_loss/Const*
	keep_dims( *
T0*

Tidx0*
_output_shapes
: 
Г
loss/dense_3_loss/truedivRealDivloss/dense_3_loss/mulloss/dense_3_loss/Mean_2*
T0*#
_output_shapes
:€€€€€€€€€
c
loss/dense_3_loss/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
Ф
loss/dense_3_loss/Mean_3Meanloss/dense_3_loss/truedivloss/dense_3_loss/Const_1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
: 
O

loss/mul/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
V
loss/mulMul
loss/mul/xloss/dense_3_loss/Mean_3*
T0*
_output_shapes
: 
В
metrics/mean_absolute_error/subSubdense_3/BiasAdddense_3_target*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€
В
metrics/mean_absolute_error/AbsAbsmetrics/mean_absolute_error/sub*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€
}
2metrics/mean_absolute_error/Mean/reduction_indicesConst*
valueB :
€€€€€€€€€*
dtype0*
_output_shapes
: 
»
 metrics/mean_absolute_error/MeanMeanmetrics/mean_absolute_error/Abs2metrics/mean_absolute_error/Mean/reduction_indices*
	keep_dims( *
T0*

Tidx0*#
_output_shapes
:€€€€€€€€€
k
!metrics/mean_absolute_error/ConstConst*
valueB: *
dtype0*
_output_shapes
:
≠
"metrics/mean_absolute_error/Mean_1Mean metrics/mean_absolute_error/Mean!metrics/mean_absolute_error/Const*
	keep_dims( *
T0*

Tidx0*
_output_shapes
: 
А
 training/RMSprop/gradients/ShapeConst*
valueB *
dtype0*
_class
loc:@loss/mul*
_output_shapes
: 
Ж
$training/RMSprop/gradients/grad_ys_0Const*
valueB
 *  А?*
dtype0*
_class
loc:@loss/mul*
_output_shapes
: 
≠
training/RMSprop/gradients/FillFill training/RMSprop/gradients/Shape$training/RMSprop/gradients/grad_ys_0*
T0*
_class
loc:@loss/mul*
_output_shapes
: 
ђ
,training/RMSprop/gradients/loss/mul_grad/MulMultraining/RMSprop/gradients/Fillloss/dense_3_loss/Mean_3*
T0*
_class
loc:@loss/mul*
_output_shapes
: 
†
.training/RMSprop/gradients/loss/mul_grad/Mul_1Multraining/RMSprop/gradients/Fill
loss/mul/x*
T0*
_class
loc:@loss/mul*
_output_shapes
: 
љ
Ftraining/RMSprop/gradients/loss/dense_3_loss/Mean_3_grad/Reshape/shapeConst*
valueB:*
dtype0*+
_class!
loc:@loss/dense_3_loss/Mean_3*
_output_shapes
:
£
@training/RMSprop/gradients/loss/dense_3_loss/Mean_3_grad/ReshapeReshape.training/RMSprop/gradients/loss/mul_grad/Mul_1Ftraining/RMSprop/gradients/loss/dense_3_loss/Mean_3_grad/Reshape/shape*
T0*
Tshape0*+
_class!
loc:@loss/dense_3_loss/Mean_3*
_output_shapes
:
ƒ
>training/RMSprop/gradients/loss/dense_3_loss/Mean_3_grad/ShapeShapeloss/dense_3_loss/truediv*
T0*
out_type0*+
_class!
loc:@loss/dense_3_loss/Mean_3*
_output_shapes
:
і
=training/RMSprop/gradients/loss/dense_3_loss/Mean_3_grad/TileTile@training/RMSprop/gradients/loss/dense_3_loss/Mean_3_grad/Reshape>training/RMSprop/gradients/loss/dense_3_loss/Mean_3_grad/Shape*
T0*

Tmultiples0*+
_class!
loc:@loss/dense_3_loss/Mean_3*#
_output_shapes
:€€€€€€€€€
∆
@training/RMSprop/gradients/loss/dense_3_loss/Mean_3_grad/Shape_1Shapeloss/dense_3_loss/truediv*
T0*
out_type0*+
_class!
loc:@loss/dense_3_loss/Mean_3*
_output_shapes
:
∞
@training/RMSprop/gradients/loss/dense_3_loss/Mean_3_grad/Shape_2Const*
valueB *
dtype0*+
_class!
loc:@loss/dense_3_loss/Mean_3*
_output_shapes
: 
µ
>training/RMSprop/gradients/loss/dense_3_loss/Mean_3_grad/ConstConst*
valueB: *
dtype0*+
_class!
loc:@loss/dense_3_loss/Mean_3*
_output_shapes
:
≤
=training/RMSprop/gradients/loss/dense_3_loss/Mean_3_grad/ProdProd@training/RMSprop/gradients/loss/dense_3_loss/Mean_3_grad/Shape_1>training/RMSprop/gradients/loss/dense_3_loss/Mean_3_grad/Const*
	keep_dims( *
T0*

Tidx0*+
_class!
loc:@loss/dense_3_loss/Mean_3*
_output_shapes
: 
Ј
@training/RMSprop/gradients/loss/dense_3_loss/Mean_3_grad/Const_1Const*
valueB: *
dtype0*+
_class!
loc:@loss/dense_3_loss/Mean_3*
_output_shapes
:
ґ
?training/RMSprop/gradients/loss/dense_3_loss/Mean_3_grad/Prod_1Prod@training/RMSprop/gradients/loss/dense_3_loss/Mean_3_grad/Shape_2@training/RMSprop/gradients/loss/dense_3_loss/Mean_3_grad/Const_1*
	keep_dims( *
T0*

Tidx0*+
_class!
loc:@loss/dense_3_loss/Mean_3*
_output_shapes
: 
±
Btraining/RMSprop/gradients/loss/dense_3_loss/Mean_3_grad/Maximum/yConst*
value	B :*
dtype0*+
_class!
loc:@loss/dense_3_loss/Mean_3*
_output_shapes
: 
Ю
@training/RMSprop/gradients/loss/dense_3_loss/Mean_3_grad/MaximumMaximum?training/RMSprop/gradients/loss/dense_3_loss/Mean_3_grad/Prod_1Btraining/RMSprop/gradients/loss/dense_3_loss/Mean_3_grad/Maximum/y*
T0*+
_class!
loc:@loss/dense_3_loss/Mean_3*
_output_shapes
: 
Ь
Atraining/RMSprop/gradients/loss/dense_3_loss/Mean_3_grad/floordivFloorDiv=training/RMSprop/gradients/loss/dense_3_loss/Mean_3_grad/Prod@training/RMSprop/gradients/loss/dense_3_loss/Mean_3_grad/Maximum*
T0*+
_class!
loc:@loss/dense_3_loss/Mean_3*
_output_shapes
: 
е
=training/RMSprop/gradients/loss/dense_3_loss/Mean_3_grad/CastCastAtraining/RMSprop/gradients/loss/dense_3_loss/Mean_3_grad/floordiv*

SrcT0*

DstT0*+
_class!
loc:@loss/dense_3_loss/Mean_3*
_output_shapes
: 
§
@training/RMSprop/gradients/loss/dense_3_loss/Mean_3_grad/truedivRealDiv=training/RMSprop/gradients/loss/dense_3_loss/Mean_3_grad/Tile=training/RMSprop/gradients/loss/dense_3_loss/Mean_3_grad/Cast*
T0*+
_class!
loc:@loss/dense_3_loss/Mean_3*#
_output_shapes
:€€€€€€€€€
¬
?training/RMSprop/gradients/loss/dense_3_loss/truediv_grad/ShapeShapeloss/dense_3_loss/mul*
T0*
out_type0*,
_class"
 loc:@loss/dense_3_loss/truediv*
_output_shapes
:
≤
Atraining/RMSprop/gradients/loss/dense_3_loss/truediv_grad/Shape_1Const*
valueB *
dtype0*,
_class"
 loc:@loss/dense_3_loss/truediv*
_output_shapes
: 
„
Otraining/RMSprop/gradients/loss/dense_3_loss/truediv_grad/BroadcastGradientArgsBroadcastGradientArgs?training/RMSprop/gradients/loss/dense_3_loss/truediv_grad/ShapeAtraining/RMSprop/gradients/loss/dense_3_loss/truediv_grad/Shape_1*
T0*,
_class"
 loc:@loss/dense_3_loss/truediv*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
Д
Atraining/RMSprop/gradients/loss/dense_3_loss/truediv_grad/RealDivRealDiv@training/RMSprop/gradients/loss/dense_3_loss/Mean_3_grad/truedivloss/dense_3_loss/Mean_2*
T0*,
_class"
 loc:@loss/dense_3_loss/truediv*#
_output_shapes
:€€€€€€€€€
∆
=training/RMSprop/gradients/loss/dense_3_loss/truediv_grad/SumSumAtraining/RMSprop/gradients/loss/dense_3_loss/truediv_grad/RealDivOtraining/RMSprop/gradients/loss/dense_3_loss/truediv_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*,
_class"
 loc:@loss/dense_3_loss/truediv*
_output_shapes
:
ґ
Atraining/RMSprop/gradients/loss/dense_3_loss/truediv_grad/ReshapeReshape=training/RMSprop/gradients/loss/dense_3_loss/truediv_grad/Sum?training/RMSprop/gradients/loss/dense_3_loss/truediv_grad/Shape*
T0*
Tshape0*,
_class"
 loc:@loss/dense_3_loss/truediv*#
_output_shapes
:€€€€€€€€€
Ј
=training/RMSprop/gradients/loss/dense_3_loss/truediv_grad/NegNegloss/dense_3_loss/mul*
T0*,
_class"
 loc:@loss/dense_3_loss/truediv*#
_output_shapes
:€€€€€€€€€
Г
Ctraining/RMSprop/gradients/loss/dense_3_loss/truediv_grad/RealDiv_1RealDiv=training/RMSprop/gradients/loss/dense_3_loss/truediv_grad/Negloss/dense_3_loss/Mean_2*
T0*,
_class"
 loc:@loss/dense_3_loss/truediv*#
_output_shapes
:€€€€€€€€€
Й
Ctraining/RMSprop/gradients/loss/dense_3_loss/truediv_grad/RealDiv_2RealDivCtraining/RMSprop/gradients/loss/dense_3_loss/truediv_grad/RealDiv_1loss/dense_3_loss/Mean_2*
T0*,
_class"
 loc:@loss/dense_3_loss/truediv*#
_output_shapes
:€€€€€€€€€
І
=training/RMSprop/gradients/loss/dense_3_loss/truediv_grad/mulMul@training/RMSprop/gradients/loss/dense_3_loss/Mean_3_grad/truedivCtraining/RMSprop/gradients/loss/dense_3_loss/truediv_grad/RealDiv_2*
T0*,
_class"
 loc:@loss/dense_3_loss/truediv*#
_output_shapes
:€€€€€€€€€
∆
?training/RMSprop/gradients/loss/dense_3_loss/truediv_grad/Sum_1Sum=training/RMSprop/gradients/loss/dense_3_loss/truediv_grad/mulQtraining/RMSprop/gradients/loss/dense_3_loss/truediv_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*,
_class"
 loc:@loss/dense_3_loss/truediv*
_output_shapes
:
ѓ
Ctraining/RMSprop/gradients/loss/dense_3_loss/truediv_grad/Reshape_1Reshape?training/RMSprop/gradients/loss/dense_3_loss/truediv_grad/Sum_1Atraining/RMSprop/gradients/loss/dense_3_loss/truediv_grad/Shape_1*
T0*
Tshape0*,
_class"
 loc:@loss/dense_3_loss/truediv*
_output_shapes
: 
љ
;training/RMSprop/gradients/loss/dense_3_loss/mul_grad/ShapeShapeloss/dense_3_loss/Mean_1*
T0*
out_type0*(
_class
loc:@loss/dense_3_loss/mul*
_output_shapes
:
љ
=training/RMSprop/gradients/loss/dense_3_loss/mul_grad/Shape_1Shapedense_3_sample_weights*
T0*
out_type0*(
_class
loc:@loss/dense_3_loss/mul*
_output_shapes
:
«
Ktraining/RMSprop/gradients/loss/dense_3_loss/mul_grad/BroadcastGradientArgsBroadcastGradientArgs;training/RMSprop/gradients/loss/dense_3_loss/mul_grad/Shape=training/RMSprop/gradients/loss/dense_3_loss/mul_grad/Shape_1*
T0*(
_class
loc:@loss/dense_3_loss/mul*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
у
9training/RMSprop/gradients/loss/dense_3_loss/mul_grad/mulMulAtraining/RMSprop/gradients/loss/dense_3_loss/truediv_grad/Reshapedense_3_sample_weights*
T0*(
_class
loc:@loss/dense_3_loss/mul*#
_output_shapes
:€€€€€€€€€
≤
9training/RMSprop/gradients/loss/dense_3_loss/mul_grad/SumSum9training/RMSprop/gradients/loss/dense_3_loss/mul_grad/mulKtraining/RMSprop/gradients/loss/dense_3_loss/mul_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*(
_class
loc:@loss/dense_3_loss/mul*
_output_shapes
:
¶
=training/RMSprop/gradients/loss/dense_3_loss/mul_grad/ReshapeReshape9training/RMSprop/gradients/loss/dense_3_loss/mul_grad/Sum;training/RMSprop/gradients/loss/dense_3_loss/mul_grad/Shape*
T0*
Tshape0*(
_class
loc:@loss/dense_3_loss/mul*#
_output_shapes
:€€€€€€€€€
ч
;training/RMSprop/gradients/loss/dense_3_loss/mul_grad/mul_1Mulloss/dense_3_loss/Mean_1Atraining/RMSprop/gradients/loss/dense_3_loss/truediv_grad/Reshape*
T0*(
_class
loc:@loss/dense_3_loss/mul*#
_output_shapes
:€€€€€€€€€
Є
;training/RMSprop/gradients/loss/dense_3_loss/mul_grad/Sum_1Sum;training/RMSprop/gradients/loss/dense_3_loss/mul_grad/mul_1Mtraining/RMSprop/gradients/loss/dense_3_loss/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*(
_class
loc:@loss/dense_3_loss/mul*
_output_shapes
:
ђ
?training/RMSprop/gradients/loss/dense_3_loss/mul_grad/Reshape_1Reshape;training/RMSprop/gradients/loss/dense_3_loss/mul_grad/Sum_1=training/RMSprop/gradients/loss/dense_3_loss/mul_grad/Shape_1*
T0*
Tshape0*(
_class
loc:@loss/dense_3_loss/mul*#
_output_shapes
:€€€€€€€€€
Ѕ
>training/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/ShapeShapeloss/dense_3_loss/Mean*
T0*
out_type0*+
_class!
loc:@loss/dense_3_loss/Mean_1*
_output_shapes
:
ђ
=training/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/SizeConst*
value	B :*
dtype0*+
_class!
loc:@loss/dense_3_loss/Mean_1*
_output_shapes
: 
ю
<training/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/addAdd*loss/dense_3_loss/Mean_1/reduction_indices=training/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/Size*
T0*+
_class!
loc:@loss/dense_3_loss/Mean_1*
_output_shapes
: 
Х
<training/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/modFloorMod<training/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/add=training/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/Size*
T0*+
_class!
loc:@loss/dense_3_loss/Mean_1*
_output_shapes
: 
Ј
@training/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/Shape_1Const*
valueB: *
dtype0*+
_class!
loc:@loss/dense_3_loss/Mean_1*
_output_shapes
:
≥
Dtraining/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/range/startConst*
value	B : *
dtype0*+
_class!
loc:@loss/dense_3_loss/Mean_1*
_output_shapes
: 
≥
Dtraining/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/range/deltaConst*
value	B :*
dtype0*+
_class!
loc:@loss/dense_3_loss/Mean_1*
_output_shapes
: 
з
>training/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/rangeRangeDtraining/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/range/start=training/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/SizeDtraining/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/range/delta*

Tidx0*+
_class!
loc:@loss/dense_3_loss/Mean_1*
_output_shapes
:
≤
Ctraining/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/Fill/valueConst*
value	B :*
dtype0*+
_class!
loc:@loss/dense_3_loss/Mean_1*
_output_shapes
: 
Ь
=training/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/FillFill@training/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/Shape_1Ctraining/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/Fill/value*
T0*+
_class!
loc:@loss/dense_3_loss/Mean_1*
_output_shapes
: 
Є
Ftraining/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/DynamicStitchDynamicStitch>training/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/range<training/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/mod>training/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/Shape=training/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/Fill*
N*
T0*+
_class!
loc:@loss/dense_3_loss/Mean_1*#
_output_shapes
:€€€€€€€€€
±
Btraining/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/Maximum/yConst*
value	B :*
dtype0*+
_class!
loc:@loss/dense_3_loss/Mean_1*
_output_shapes
: 
≤
@training/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/MaximumMaximumFtraining/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/DynamicStitchBtraining/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/Maximum/y*
T0*+
_class!
loc:@loss/dense_3_loss/Mean_1*#
_output_shapes
:€€€€€€€€€
™
Atraining/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/floordivFloorDiv>training/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/Shape@training/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/Maximum*
T0*+
_class!
loc:@loss/dense_3_loss/Mean_1*#
_output_shapes
:€€€€€€€€€
∞
@training/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/ReshapeReshape=training/RMSprop/gradients/loss/dense_3_loss/mul_grad/ReshapeFtraining/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/DynamicStitch*
T0*
Tshape0*+
_class!
loc:@loss/dense_3_loss/Mean_1*
_output_shapes
:
ђ
=training/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/TileTile@training/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/ReshapeAtraining/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/floordiv*
T0*

Tmultiples0*+
_class!
loc:@loss/dense_3_loss/Mean_1*
_output_shapes
:
√
@training/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/Shape_2Shapeloss/dense_3_loss/Mean*
T0*
out_type0*+
_class!
loc:@loss/dense_3_loss/Mean_1*
_output_shapes
:
≈
@training/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/Shape_3Shapeloss/dense_3_loss/Mean_1*
T0*
out_type0*+
_class!
loc:@loss/dense_3_loss/Mean_1*
_output_shapes
:
µ
>training/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/ConstConst*
valueB: *
dtype0*+
_class!
loc:@loss/dense_3_loss/Mean_1*
_output_shapes
:
≤
=training/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/ProdProd@training/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/Shape_2>training/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/Const*
	keep_dims( *
T0*

Tidx0*+
_class!
loc:@loss/dense_3_loss/Mean_1*
_output_shapes
: 
Ј
@training/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/Const_1Const*
valueB: *
dtype0*+
_class!
loc:@loss/dense_3_loss/Mean_1*
_output_shapes
:
ґ
?training/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/Prod_1Prod@training/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/Shape_3@training/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/Const_1*
	keep_dims( *
T0*

Tidx0*+
_class!
loc:@loss/dense_3_loss/Mean_1*
_output_shapes
: 
≥
Dtraining/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/Maximum_1/yConst*
value	B :*
dtype0*+
_class!
loc:@loss/dense_3_loss/Mean_1*
_output_shapes
: 
Ґ
Btraining/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/Maximum_1Maximum?training/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/Prod_1Dtraining/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/Maximum_1/y*
T0*+
_class!
loc:@loss/dense_3_loss/Mean_1*
_output_shapes
: 
†
Ctraining/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/floordiv_1FloorDiv=training/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/ProdBtraining/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/Maximum_1*
T0*+
_class!
loc:@loss/dense_3_loss/Mean_1*
_output_shapes
: 
з
=training/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/CastCastCtraining/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/floordiv_1*

SrcT0*

DstT0*+
_class!
loc:@loss/dense_3_loss/Mean_1*
_output_shapes
: 
§
@training/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/truedivRealDiv=training/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/Tile=training/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/Cast*
T0*+
_class!
loc:@loss/dense_3_loss/Mean_1*#
_output_shapes
:€€€€€€€€€
њ
<training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/ShapeShapeloss/dense_3_loss/Square*
T0*
out_type0*)
_class
loc:@loss/dense_3_loss/Mean*
_output_shapes
:
®
;training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/SizeConst*
value	B :*
dtype0*)
_class
loc:@loss/dense_3_loss/Mean*
_output_shapes
: 
ф
:training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/addAdd(loss/dense_3_loss/Mean/reduction_indices;training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/Size*
T0*)
_class
loc:@loss/dense_3_loss/Mean*
_output_shapes
: 
Л
:training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/modFloorMod:training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/add;training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/Size*
T0*)
_class
loc:@loss/dense_3_loss/Mean*
_output_shapes
: 
ђ
>training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/Shape_1Const*
valueB *
dtype0*)
_class
loc:@loss/dense_3_loss/Mean*
_output_shapes
: 
ѓ
Btraining/RMSprop/gradients/loss/dense_3_loss/Mean_grad/range/startConst*
value	B : *
dtype0*)
_class
loc:@loss/dense_3_loss/Mean*
_output_shapes
: 
ѓ
Btraining/RMSprop/gradients/loss/dense_3_loss/Mean_grad/range/deltaConst*
value	B :*
dtype0*)
_class
loc:@loss/dense_3_loss/Mean*
_output_shapes
: 
Ё
<training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/rangeRangeBtraining/RMSprop/gradients/loss/dense_3_loss/Mean_grad/range/start;training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/SizeBtraining/RMSprop/gradients/loss/dense_3_loss/Mean_grad/range/delta*

Tidx0*)
_class
loc:@loss/dense_3_loss/Mean*
_output_shapes
:
Ѓ
Atraining/RMSprop/gradients/loss/dense_3_loss/Mean_grad/Fill/valueConst*
value	B :*
dtype0*)
_class
loc:@loss/dense_3_loss/Mean*
_output_shapes
: 
Т
;training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/FillFill>training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/Shape_1Atraining/RMSprop/gradients/loss/dense_3_loss/Mean_grad/Fill/value*
T0*)
_class
loc:@loss/dense_3_loss/Mean*
_output_shapes
: 
ђ
Dtraining/RMSprop/gradients/loss/dense_3_loss/Mean_grad/DynamicStitchDynamicStitch<training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/range:training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/mod<training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/Shape;training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/Fill*
N*
T0*)
_class
loc:@loss/dense_3_loss/Mean*#
_output_shapes
:€€€€€€€€€
≠
@training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/Maximum/yConst*
value	B :*
dtype0*)
_class
loc:@loss/dense_3_loss/Mean*
_output_shapes
: 
™
>training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/MaximumMaximumDtraining/RMSprop/gradients/loss/dense_3_loss/Mean_grad/DynamicStitch@training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/Maximum/y*
T0*)
_class
loc:@loss/dense_3_loss/Mean*#
_output_shapes
:€€€€€€€€€
Щ
?training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/floordivFloorDiv<training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/Shape>training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/Maximum*
T0*)
_class
loc:@loss/dense_3_loss/Mean*
_output_shapes
:
≠
>training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/ReshapeReshape@training/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/truedivDtraining/RMSprop/gradients/loss/dense_3_loss/Mean_grad/DynamicStitch*
T0*
Tshape0*)
_class
loc:@loss/dense_3_loss/Mean*
_output_shapes
:
Љ
;training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/TileTile>training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/Reshape?training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/floordiv*
T0*

Tmultiples0*)
_class
loc:@loss/dense_3_loss/Mean*0
_output_shapes
:€€€€€€€€€€€€€€€€€€
Ѕ
>training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/Shape_2Shapeloss/dense_3_loss/Square*
T0*
out_type0*)
_class
loc:@loss/dense_3_loss/Mean*
_output_shapes
:
њ
>training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/Shape_3Shapeloss/dense_3_loss/Mean*
T0*
out_type0*)
_class
loc:@loss/dense_3_loss/Mean*
_output_shapes
:
±
<training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/ConstConst*
valueB: *
dtype0*)
_class
loc:@loss/dense_3_loss/Mean*
_output_shapes
:
™
;training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/ProdProd>training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/Shape_2<training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/Const*
	keep_dims( *
T0*

Tidx0*)
_class
loc:@loss/dense_3_loss/Mean*
_output_shapes
: 
≥
>training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/Const_1Const*
valueB: *
dtype0*)
_class
loc:@loss/dense_3_loss/Mean*
_output_shapes
:
Ѓ
=training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/Prod_1Prod>training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/Shape_3>training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/Const_1*
	keep_dims( *
T0*

Tidx0*)
_class
loc:@loss/dense_3_loss/Mean*
_output_shapes
: 
ѓ
Btraining/RMSprop/gradients/loss/dense_3_loss/Mean_grad/Maximum_1/yConst*
value	B :*
dtype0*)
_class
loc:@loss/dense_3_loss/Mean*
_output_shapes
: 
Ъ
@training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/Maximum_1Maximum=training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/Prod_1Btraining/RMSprop/gradients/loss/dense_3_loss/Mean_grad/Maximum_1/y*
T0*)
_class
loc:@loss/dense_3_loss/Mean*
_output_shapes
: 
Ш
Atraining/RMSprop/gradients/loss/dense_3_loss/Mean_grad/floordiv_1FloorDiv;training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/Prod@training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/Maximum_1*
T0*)
_class
loc:@loss/dense_3_loss/Mean*
_output_shapes
: 
б
;training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/CastCastAtraining/RMSprop/gradients/loss/dense_3_loss/Mean_grad/floordiv_1*

SrcT0*

DstT0*)
_class
loc:@loss/dense_3_loss/Mean*
_output_shapes
: 
©
>training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/truedivRealDiv;training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/Tile;training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/Cast*
T0*)
_class
loc:@loss/dense_3_loss/Mean*0
_output_shapes
:€€€€€€€€€€€€€€€€€€
с
>training/RMSprop/gradients/loss/dense_3_loss/Square_grad/mul/xConst?^training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/truediv*
valueB
 *   @*
dtype0*+
_class!
loc:@loss/dense_3_loss/Square*
_output_shapes
: 
В
<training/RMSprop/gradients/loss/dense_3_loss/Square_grad/mulMul>training/RMSprop/gradients/loss/dense_3_loss/Square_grad/mul/xloss/dense_3_loss/sub*
T0*+
_class!
loc:@loss/dense_3_loss/Square*0
_output_shapes
:€€€€€€€€€€€€€€€€€€
Ђ
>training/RMSprop/gradients/loss/dense_3_loss/Square_grad/mul_1Mul>training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/truediv<training/RMSprop/gradients/loss/dense_3_loss/Square_grad/mul*
T0*+
_class!
loc:@loss/dense_3_loss/Square*0
_output_shapes
:€€€€€€€€€€€€€€€€€€
і
;training/RMSprop/gradients/loss/dense_3_loss/sub_grad/ShapeShapedense_3/BiasAdd*
T0*
out_type0*(
_class
loc:@loss/dense_3_loss/sub*
_output_shapes
:
µ
=training/RMSprop/gradients/loss/dense_3_loss/sub_grad/Shape_1Shapedense_3_target*
T0*
out_type0*(
_class
loc:@loss/dense_3_loss/sub*
_output_shapes
:
«
Ktraining/RMSprop/gradients/loss/dense_3_loss/sub_grad/BroadcastGradientArgsBroadcastGradientArgs;training/RMSprop/gradients/loss/dense_3_loss/sub_grad/Shape=training/RMSprop/gradients/loss/dense_3_loss/sub_grad/Shape_1*
T0*(
_class
loc:@loss/dense_3_loss/sub*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
Ј
9training/RMSprop/gradients/loss/dense_3_loss/sub_grad/SumSum>training/RMSprop/gradients/loss/dense_3_loss/Square_grad/mul_1Ktraining/RMSprop/gradients/loss/dense_3_loss/sub_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*(
_class
loc:@loss/dense_3_loss/sub*
_output_shapes
:
™
=training/RMSprop/gradients/loss/dense_3_loss/sub_grad/ReshapeReshape9training/RMSprop/gradients/loss/dense_3_loss/sub_grad/Sum;training/RMSprop/gradients/loss/dense_3_loss/sub_grad/Shape*
T0*
Tshape0*(
_class
loc:@loss/dense_3_loss/sub*'
_output_shapes
:€€€€€€€€€
ї
;training/RMSprop/gradients/loss/dense_3_loss/sub_grad/Sum_1Sum>training/RMSprop/gradients/loss/dense_3_loss/Square_grad/mul_1Mtraining/RMSprop/gradients/loss/dense_3_loss/sub_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*(
_class
loc:@loss/dense_3_loss/sub*
_output_shapes
:
 
9training/RMSprop/gradients/loss/dense_3_loss/sub_grad/NegNeg;training/RMSprop/gradients/loss/dense_3_loss/sub_grad/Sum_1*
T0*(
_class
loc:@loss/dense_3_loss/sub*
_output_shapes
:
Ј
?training/RMSprop/gradients/loss/dense_3_loss/sub_grad/Reshape_1Reshape9training/RMSprop/gradients/loss/dense_3_loss/sub_grad/Neg=training/RMSprop/gradients/loss/dense_3_loss/sub_grad/Shape_1*
T0*
Tshape0*(
_class
loc:@loss/dense_3_loss/sub*0
_output_shapes
:€€€€€€€€€€€€€€€€€€
й
;training/RMSprop/gradients/dense_3/BiasAdd_grad/BiasAddGradBiasAddGrad=training/RMSprop/gradients/loss/dense_3_loss/sub_grad/Reshape*
T0*
data_formatNHWC*"
_class
loc:@dense_3/BiasAdd*
_output_shapes
:
О
5training/RMSprop/gradients/dense_3/MatMul_grad/MatMulMatMul=training/RMSprop/gradients/loss/dense_3_loss/sub_grad/Reshapedense_3/kernel/read*
transpose_a( *
transpose_b(*
T0*!
_class
loc:@dense_3/MatMul*'
_output_shapes
:€€€€€€€€€@
А
7training/RMSprop/gradients/dense_3/MatMul_grad/MatMul_1MatMuldense_2/Relu=training/RMSprop/gradients/loss/dense_3_loss/sub_grad/Reshape*
transpose_a(*
transpose_b( *
T0*!
_class
loc:@dense_3/MatMul*
_output_shapes

:@
ў
5training/RMSprop/gradients/dense_2/Relu_grad/ReluGradReluGrad5training/RMSprop/gradients/dense_3/MatMul_grad/MatMuldense_2/Relu*
T0*
_class
loc:@dense_2/Relu*'
_output_shapes
:€€€€€€€€€@
б
;training/RMSprop/gradients/dense_2/BiasAdd_grad/BiasAddGradBiasAddGrad5training/RMSprop/gradients/dense_2/Relu_grad/ReluGrad*
T0*
data_formatNHWC*"
_class
loc:@dense_2/BiasAdd*
_output_shapes
:@
Ж
5training/RMSprop/gradients/dense_2/MatMul_grad/MatMulMatMul5training/RMSprop/gradients/dense_2/Relu_grad/ReluGraddense_2/kernel/read*
transpose_a( *
transpose_b(*
T0*!
_class
loc:@dense_2/MatMul*'
_output_shapes
:€€€€€€€€€@
ш
7training/RMSprop/gradients/dense_2/MatMul_grad/MatMul_1MatMuldense_1/Relu5training/RMSprop/gradients/dense_2/Relu_grad/ReluGrad*
transpose_a(*
transpose_b( *
T0*!
_class
loc:@dense_2/MatMul*
_output_shapes

:@@
ў
5training/RMSprop/gradients/dense_1/Relu_grad/ReluGradReluGrad5training/RMSprop/gradients/dense_2/MatMul_grad/MatMuldense_1/Relu*
T0*
_class
loc:@dense_1/Relu*'
_output_shapes
:€€€€€€€€€@
б
;training/RMSprop/gradients/dense_1/BiasAdd_grad/BiasAddGradBiasAddGrad5training/RMSprop/gradients/dense_1/Relu_grad/ReluGrad*
T0*
data_formatNHWC*"
_class
loc:@dense_1/BiasAdd*
_output_shapes
:@
Ж
5training/RMSprop/gradients/dense_1/MatMul_grad/MatMulMatMul5training/RMSprop/gradients/dense_1/Relu_grad/ReluGraddense_1/kernel/read*
transpose_a( *
transpose_b(*
T0*!
_class
loc:@dense_1/MatMul*'
_output_shapes
:€€€€€€€€€
щ
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
Н
training/RMSprop/Variable
VariableV2*
shape
:@*
dtype0*
	container *
shared_name *
_output_shapes

:@
Ё
 training/RMSprop/Variable/AssignAssigntraining/RMSprop/Variabletraining/RMSprop/Const*
T0*
validate_shape(*
use_locking(*,
_class"
 loc:@training/RMSprop/Variable*
_output_shapes

:@
Ь
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
З
training/RMSprop/Variable_1
VariableV2*
shape:@*
dtype0*
	container *
shared_name *
_output_shapes
:@
б
"training/RMSprop/Variable_1/AssignAssigntraining/RMSprop/Variable_1training/RMSprop/Const_1*
T0*
validate_shape(*
use_locking(*.
_class$
" loc:@training/RMSprop/Variable_1*
_output_shapes
:@
Ю
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
П
training/RMSprop/Variable_2
VariableV2*
shape
:@@*
dtype0*
	container *
shared_name *
_output_shapes

:@@
е
"training/RMSprop/Variable_2/AssignAssigntraining/RMSprop/Variable_2training/RMSprop/Const_2*
T0*
validate_shape(*
use_locking(*.
_class$
" loc:@training/RMSprop/Variable_2*
_output_shapes

:@@
Ґ
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
З
training/RMSprop/Variable_3
VariableV2*
shape:@*
dtype0*
	container *
shared_name *
_output_shapes
:@
б
"training/RMSprop/Variable_3/AssignAssigntraining/RMSprop/Variable_3training/RMSprop/Const_3*
T0*
validate_shape(*
use_locking(*.
_class$
" loc:@training/RMSprop/Variable_3*
_output_shapes
:@
Ю
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
П
training/RMSprop/Variable_4
VariableV2*
shape
:@*
dtype0*
	container *
shared_name *
_output_shapes

:@
е
"training/RMSprop/Variable_4/AssignAssigntraining/RMSprop/Variable_4training/RMSprop/Const_4*
T0*
validate_shape(*
use_locking(*.
_class$
" loc:@training/RMSprop/Variable_4*
_output_shapes

:@
Ґ
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
З
training/RMSprop/Variable_5
VariableV2*
shape:*
dtype0*
	container *
shared_name *
_output_shapes
:
б
"training/RMSprop/Variable_5/AssignAssigntraining/RMSprop/Variable_5training/RMSprop/Const_5*
T0*
validate_shape(*
use_locking(*.
_class$
" loc:@training/RMSprop/Variable_5*
_output_shapes
:
Ю
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
Є
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
 *  А?*
dtype0*
_output_shapes
: 
f
training/RMSprop/subSubtraining/RMSprop/sub/xRMSprop/rho/read*
T0*
_output_shapes
: 
Г
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
“
training/RMSprop/AssignAssigntraining/RMSprop/Variabletraining/RMSprop/add*
T0*
validate_shape(*
use_locking(*,
_class"
 loc:@training/RMSprop/Variable*
_output_shapes

:@
Р
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
 *  А*
dtype0*
_output_shapes
: 
К
&training/RMSprop/clip_by_value/MinimumMinimumtraining/RMSprop/addtraining/RMSprop/Const_7*
T0*
_output_shapes

:@
Ф
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
 *wћ+2*
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
ј
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
 *  А?*
dtype0*
_output_shapes
: 
j
training/RMSprop/sub_2Subtraining/RMSprop/sub_2/xRMSprop/rho/read*
T0*
_output_shapes
: 
Е
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
÷
training/RMSprop/Assign_2Assigntraining/RMSprop/Variable_1training/RMSprop/add_2*
T0*
validate_shape(*
use_locking(*.
_class$
" loc:@training/RMSprop/Variable_1*
_output_shapes
:@
Р
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
 *  А*
dtype0*
_output_shapes
: 
К
(training/RMSprop/clip_by_value_1/MinimumMinimumtraining/RMSprop/add_2training/RMSprop/Const_9*
T0*
_output_shapes
:@
Ф
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
 *wћ+2*
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
Є
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
 *  А?*
dtype0*
_output_shapes
: 
j
training/RMSprop/sub_4Subtraining/RMSprop/sub_4/xRMSprop/rho/read*
T0*
_output_shapes
: 
Е
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
Џ
training/RMSprop/Assign_4Assigntraining/RMSprop/Variable_2training/RMSprop/add_4*
T0*
validate_shape(*
use_locking(*.
_class$
" loc:@training/RMSprop/Variable_2*
_output_shapes

:@@
Р
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
 *  А*
dtype0*
_output_shapes
: 
П
(training/RMSprop/clip_by_value_2/MinimumMinimumtraining/RMSprop/add_4training/RMSprop/Const_11*
T0*
_output_shapes

:@@
Щ
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
 *wћ+2*
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
ј
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
 *  А?*
dtype0*
_output_shapes
: 
j
training/RMSprop/sub_6Subtraining/RMSprop/sub_6/xRMSprop/rho/read*
T0*
_output_shapes
: 
Е
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
÷
training/RMSprop/Assign_6Assigntraining/RMSprop/Variable_3training/RMSprop/add_6*
T0*
validate_shape(*
use_locking(*.
_class$
" loc:@training/RMSprop/Variable_3*
_output_shapes
:@
С
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
 *  А*
dtype0*
_output_shapes
: 
Л
(training/RMSprop/clip_by_value_3/MinimumMinimumtraining/RMSprop/add_6training/RMSprop/Const_13*
T0*
_output_shapes
:@
Х
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
 *wћ+2*
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
Є
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
 *  А?*
dtype0*
_output_shapes
: 
j
training/RMSprop/sub_8Subtraining/RMSprop/sub_8/xRMSprop/rho/read*
T0*
_output_shapes
: 
Е
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
Џ
training/RMSprop/Assign_8Assigntraining/RMSprop/Variable_4training/RMSprop/add_8*
T0*
validate_shape(*
use_locking(*.
_class$
" loc:@training/RMSprop/Variable_4*
_output_shapes

:@
С
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
 *  А*
dtype0*
_output_shapes
: 
П
(training/RMSprop/clip_by_value_4/MinimumMinimumtraining/RMSprop/add_8training/RMSprop/Const_15*
T0*
_output_shapes

:@
Щ
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
 *wћ+2*
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
ј
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
 *  А?*
dtype0*
_output_shapes
: 
l
training/RMSprop/sub_10Subtraining/RMSprop/sub_10/xRMSprop/rho/read*
T0*
_output_shapes
: 
Е
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
Ў
training/RMSprop/Assign_10Assigntraining/RMSprop/Variable_5training/RMSprop/add_10*
T0*
validate_shape(*
use_locking(*.
_class$
" loc:@training/RMSprop/Variable_5*
_output_shapes
:
С
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
 *  А*
dtype0*
_output_shapes
: 
М
(training/RMSprop/clip_by_value_5/MinimumMinimumtraining/RMSprop/add_10training/RMSprop/Const_17*
T0*
_output_shapes
:
Х
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
 *wћ+2*
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
Ї
training/RMSprop/Assign_11Assigndense_3/biastraining/RMSprop/sub_11*
T0*
validate_shape(*
use_locking(*
_class
loc:@dense_3/bias*
_output_shapes
:
Є
training/group_depsNoOp	^loss/mul#^metrics/mean_absolute_error/Mean_1^training/RMSprop/AssignAdd^training/RMSprop/Assign^training/RMSprop/Assign_1^training/RMSprop/Assign_2^training/RMSprop/Assign_3^training/RMSprop/Assign_4^training/RMSprop/Assign_5^training/RMSprop/Assign_6^training/RMSprop/Assign_7^training/RMSprop/Assign_8^training/RMSprop/Assign_9^training/RMSprop/Assign_10^training/RMSprop/Assign_11
B

group_depsNoOp	^loss/mul#^metrics/mean_absolute_error/Mean_1
Ж
IsVariableInitializedIsVariableInitializeddense_1/kernel*
dtype0*!
_class
loc:@dense_1/kernel*
_output_shapes
: 
Д
IsVariableInitialized_1IsVariableInitializeddense_1/bias*
dtype0*
_class
loc:@dense_1/bias*
_output_shapes
: 
И
IsVariableInitialized_2IsVariableInitializeddense_2/kernel*
dtype0*!
_class
loc:@dense_2/kernel*
_output_shapes
: 
Д
IsVariableInitialized_3IsVariableInitializeddense_2/bias*
dtype0*
_class
loc:@dense_2/bias*
_output_shapes
: 
И
IsVariableInitialized_4IsVariableInitializeddense_3/kernel*
dtype0*!
_class
loc:@dense_3/kernel*
_output_shapes
: 
Д
IsVariableInitialized_5IsVariableInitializeddense_3/bias*
dtype0*
_class
loc:@dense_3/bias*
_output_shapes
: 
А
IsVariableInitialized_6IsVariableInitialized
RMSprop/lr*
dtype0*
_class
loc:@RMSprop/lr*
_output_shapes
: 
В
IsVariableInitialized_7IsVariableInitializedRMSprop/rho*
dtype0*
_class
loc:@RMSprop/rho*
_output_shapes
: 
Ж
IsVariableInitialized_8IsVariableInitializedRMSprop/decay*
dtype0* 
_class
loc:@RMSprop/decay*
_output_shapes
: 
Р
IsVariableInitialized_9IsVariableInitializedRMSprop/iterations*
dtype0	*%
_class
loc:@RMSprop/iterations*
_output_shapes
: 
Я
IsVariableInitialized_10IsVariableInitializedtraining/RMSprop/Variable*
dtype0*,
_class"
 loc:@training/RMSprop/Variable*
_output_shapes
: 
£
IsVariableInitialized_11IsVariableInitializedtraining/RMSprop/Variable_1*
dtype0*.
_class$
" loc:@training/RMSprop/Variable_1*
_output_shapes
: 
£
IsVariableInitialized_12IsVariableInitializedtraining/RMSprop/Variable_2*
dtype0*.
_class$
" loc:@training/RMSprop/Variable_2*
_output_shapes
: 
£
IsVariableInitialized_13IsVariableInitializedtraining/RMSprop/Variable_3*
dtype0*.
_class$
" loc:@training/RMSprop/Variable_3*
_output_shapes
: 
£
IsVariableInitialized_14IsVariableInitializedtraining/RMSprop/Variable_4*
dtype0*.
_class$
" loc:@training/RMSprop/Variable_4*
_output_shapes
: 
£
IsVariableInitialized_15IsVariableInitializedtraining/RMSprop/Variable_5*
dtype0*.
_class$
" loc:@training/RMSprop/Variable_5*
_output_shapes
: 
ќ
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
–
Merge/MergeSummaryMergeSummarydense_1/kernel_0dense_1/bias_0dense_1_outdense_2/kernel_0dense_2/bias_0dense_2_outdense_3/kernel_0dense_3/bias_0dense_3_out*
N	*
_output_shapes
: "еBI«     lД≥e	mв≈”л£÷AJРЃ
яњ
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
ref"TА

value"T

output_ref"TА"	
Ttype"
validate_shapebool("
use_lockingbool(Ш
s
	AssignAdd
ref"TА

value"T

output_ref"TА" 
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
ref"dtypeА
is_initialized
"
dtypetypeШ
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

2	Р
Н
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

2	Р
=
Mul
x"T
y"T
z"T"
Ttype:
2	Р
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
Р
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
Н
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
2	И
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
М
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
ref"dtypeА"
shapeshape"
dtypetype"
	containerstring "
shared_namestring И*1.5.02v1.5.0-0-g37aa430d84©ш
p
dense_1_inputPlaceholder*
dtype0*
shape:€€€€€€€€€*'
_output_shapes
:€€€€€€€€€
m
dense_1/random_uniform/shapeConst*
valueB"   @   *
dtype0*
_output_shapes
:
_
dense_1/random_uniform/minConst*
valueB
 *!мОЊ*
dtype0*
_output_shapes
: 
_
dense_1/random_uniform/maxConst*
valueB
 *!мО>*
dtype0*
_output_shapes
: 
®
$dense_1/random_uniform/RandomUniformRandomUniformdense_1/random_uniform/shape*
seed±€е)*
seed2ічЩ*
dtype0*
T0*
_output_shapes

:@
z
dense_1/random_uniform/subSubdense_1/random_uniform/maxdense_1/random_uniform/min*
T0*
_output_shapes
: 
М
dense_1/random_uniform/mulMul$dense_1/random_uniform/RandomUniformdense_1/random_uniform/sub*
T0*
_output_shapes

:@
~
dense_1/random_uniformAdddense_1/random_uniform/muldense_1/random_uniform/min*
T0*
_output_shapes

:@
В
dense_1/kernel
VariableV2*
shape
:@*
dtype0*
	container *
shared_name *
_output_shapes

:@
Љ
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
©
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
Ф
dense_1/MatMulMatMuldense_1_inputdense_1/kernel/read*
transpose_a( *
transpose_b( *
T0*'
_output_shapes
:€€€€€€€€€@
Ж
dense_1/BiasAddBiasAdddense_1/MatMuldense_1/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:€€€€€€€€€@
W
dense_1/ReluReludense_1/BiasAdd*
T0*'
_output_shapes
:€€€€€€€€€@
m
dense_2/random_uniform/shapeConst*
valueB"@   @   *
dtype0*
_output_shapes
:
_
dense_2/random_uniform/minConst*
valueB
 *„≥]Њ*
dtype0*
_output_shapes
: 
_
dense_2/random_uniform/maxConst*
valueB
 *„≥]>*
dtype0*
_output_shapes
: 
®
$dense_2/random_uniform/RandomUniformRandomUniformdense_2/random_uniform/shape*
seed±€е)*
seed2€єх*
dtype0*
T0*
_output_shapes

:@@
z
dense_2/random_uniform/subSubdense_2/random_uniform/maxdense_2/random_uniform/min*
T0*
_output_shapes
: 
М
dense_2/random_uniform/mulMul$dense_2/random_uniform/RandomUniformdense_2/random_uniform/sub*
T0*
_output_shapes

:@@
~
dense_2/random_uniformAdddense_2/random_uniform/muldense_2/random_uniform/min*
T0*
_output_shapes

:@@
В
dense_2/kernel
VariableV2*
shape
:@@*
dtype0*
	container *
shared_name *
_output_shapes

:@@
Љ
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
©
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
У
dense_2/MatMulMatMuldense_1/Reludense_2/kernel/read*
transpose_a( *
transpose_b( *
T0*'
_output_shapes
:€€€€€€€€€@
Ж
dense_2/BiasAddBiasAdddense_2/MatMuldense_2/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:€€€€€€€€€@
W
dense_2/ReluReludense_2/BiasAdd*
T0*'
_output_shapes
:€€€€€€€€€@
m
dense_3/random_uniform/shapeConst*
valueB"@      *
dtype0*
_output_shapes
:
_
dense_3/random_uniform/minConst*
valueB
 *ИОЫЊ*
dtype0*
_output_shapes
: 
_
dense_3/random_uniform/maxConst*
valueB
 *ИОЫ>*
dtype0*
_output_shapes
: 
®
$dense_3/random_uniform/RandomUniformRandomUniformdense_3/random_uniform/shape*
seed±€е)*
seed2ицЯ*
dtype0*
T0*
_output_shapes

:@
z
dense_3/random_uniform/subSubdense_3/random_uniform/maxdense_3/random_uniform/min*
T0*
_output_shapes
: 
М
dense_3/random_uniform/mulMul$dense_3/random_uniform/RandomUniformdense_3/random_uniform/sub*
T0*
_output_shapes

:@
~
dense_3/random_uniformAdddense_3/random_uniform/muldense_3/random_uniform/min*
T0*
_output_shapes

:@
В
dense_3/kernel
VariableV2*
shape
:@*
dtype0*
	container *
shared_name *
_output_shapes

:@
Љ
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
©
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
У
dense_3/MatMulMatMuldense_2/Reludense_3/kernel/read*
transpose_a( *
transpose_b( *
T0*'
_output_shapes
:€€€€€€€€€
Ж
dense_3/BiasAddBiasAdddense_3/MatMuldense_3/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:€€€€€€€€€
]
RMSprop/lr/initial_valueConst*
valueB
 *oГ:*
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
™
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
Ѓ
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
ґ
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
 
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
Г
dense_3_targetPlaceholder*
dtype0*%
shape:€€€€€€€€€€€€€€€€€€*0
_output_shapes
:€€€€€€€€€€€€€€€€€€
q
dense_3_sample_weightsPlaceholder*
dtype0*
shape:€€€€€€€€€*#
_output_shapes
:€€€€€€€€€
x
loss/dense_3_loss/subSubdense_3/BiasAdddense_3_target*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€
t
loss/dense_3_loss/SquareSquareloss/dense_3_loss/sub*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€
s
(loss/dense_3_loss/Mean/reduction_indicesConst*
valueB :
€€€€€€€€€*
dtype0*
_output_shapes
: 
≠
loss/dense_3_loss/MeanMeanloss/dense_3_loss/Square(loss/dense_3_loss/Mean/reduction_indices*
	keep_dims( *
T0*

Tidx0*#
_output_shapes
:€€€€€€€€€
m
*loss/dense_3_loss/Mean_1/reduction_indicesConst*
valueB *
dtype0*
_output_shapes
: 
ѓ
loss/dense_3_loss/Mean_1Meanloss/dense_3_loss/Mean*loss/dense_3_loss/Mean_1/reduction_indices*
	keep_dims( *
T0*

Tidx0*#
_output_shapes
:€€€€€€€€€
|
loss/dense_3_loss/mulMulloss/dense_3_loss/Mean_1dense_3_sample_weights*
T0*#
_output_shapes
:€€€€€€€€€
a
loss/dense_3_loss/NotEqual/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
К
loss/dense_3_loss/NotEqualNotEqualdense_3_sample_weightsloss/dense_3_loss/NotEqual/y*
T0*#
_output_shapes
:€€€€€€€€€
w
loss/dense_3_loss/CastCastloss/dense_3_loss/NotEqual*

SrcT0
*

DstT0*#
_output_shapes
:€€€€€€€€€
a
loss/dense_3_loss/ConstConst*
valueB: *
dtype0*
_output_shapes
:
П
loss/dense_3_loss/Mean_2Meanloss/dense_3_loss/Castloss/dense_3_loss/Const*
	keep_dims( *
T0*

Tidx0*
_output_shapes
: 
Г
loss/dense_3_loss/truedivRealDivloss/dense_3_loss/mulloss/dense_3_loss/Mean_2*
T0*#
_output_shapes
:€€€€€€€€€
c
loss/dense_3_loss/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
Ф
loss/dense_3_loss/Mean_3Meanloss/dense_3_loss/truedivloss/dense_3_loss/Const_1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
: 
O

loss/mul/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
V
loss/mulMul
loss/mul/xloss/dense_3_loss/Mean_3*
T0*
_output_shapes
: 
В
metrics/mean_absolute_error/subSubdense_3/BiasAdddense_3_target*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€
В
metrics/mean_absolute_error/AbsAbsmetrics/mean_absolute_error/sub*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€
}
2metrics/mean_absolute_error/Mean/reduction_indicesConst*
valueB :
€€€€€€€€€*
dtype0*
_output_shapes
: 
»
 metrics/mean_absolute_error/MeanMeanmetrics/mean_absolute_error/Abs2metrics/mean_absolute_error/Mean/reduction_indices*
	keep_dims( *
T0*

Tidx0*#
_output_shapes
:€€€€€€€€€
k
!metrics/mean_absolute_error/ConstConst*
valueB: *
dtype0*
_output_shapes
:
≠
"metrics/mean_absolute_error/Mean_1Mean metrics/mean_absolute_error/Mean!metrics/mean_absolute_error/Const*
	keep_dims( *
T0*

Tidx0*
_output_shapes
: 
А
 training/RMSprop/gradients/ShapeConst*
valueB *
dtype0*
_class
loc:@loss/mul*
_output_shapes
: 
Ж
$training/RMSprop/gradients/grad_ys_0Const*
valueB
 *  А?*
dtype0*
_class
loc:@loss/mul*
_output_shapes
: 
≠
training/RMSprop/gradients/FillFill training/RMSprop/gradients/Shape$training/RMSprop/gradients/grad_ys_0*
T0*
_class
loc:@loss/mul*
_output_shapes
: 
ђ
,training/RMSprop/gradients/loss/mul_grad/MulMultraining/RMSprop/gradients/Fillloss/dense_3_loss/Mean_3*
T0*
_class
loc:@loss/mul*
_output_shapes
: 
†
.training/RMSprop/gradients/loss/mul_grad/Mul_1Multraining/RMSprop/gradients/Fill
loss/mul/x*
T0*
_class
loc:@loss/mul*
_output_shapes
: 
љ
Ftraining/RMSprop/gradients/loss/dense_3_loss/Mean_3_grad/Reshape/shapeConst*
valueB:*
dtype0*+
_class!
loc:@loss/dense_3_loss/Mean_3*
_output_shapes
:
£
@training/RMSprop/gradients/loss/dense_3_loss/Mean_3_grad/ReshapeReshape.training/RMSprop/gradients/loss/mul_grad/Mul_1Ftraining/RMSprop/gradients/loss/dense_3_loss/Mean_3_grad/Reshape/shape*
T0*
Tshape0*+
_class!
loc:@loss/dense_3_loss/Mean_3*
_output_shapes
:
ƒ
>training/RMSprop/gradients/loss/dense_3_loss/Mean_3_grad/ShapeShapeloss/dense_3_loss/truediv*
T0*
out_type0*+
_class!
loc:@loss/dense_3_loss/Mean_3*
_output_shapes
:
і
=training/RMSprop/gradients/loss/dense_3_loss/Mean_3_grad/TileTile@training/RMSprop/gradients/loss/dense_3_loss/Mean_3_grad/Reshape>training/RMSprop/gradients/loss/dense_3_loss/Mean_3_grad/Shape*
T0*

Tmultiples0*+
_class!
loc:@loss/dense_3_loss/Mean_3*#
_output_shapes
:€€€€€€€€€
∆
@training/RMSprop/gradients/loss/dense_3_loss/Mean_3_grad/Shape_1Shapeloss/dense_3_loss/truediv*
T0*
out_type0*+
_class!
loc:@loss/dense_3_loss/Mean_3*
_output_shapes
:
∞
@training/RMSprop/gradients/loss/dense_3_loss/Mean_3_grad/Shape_2Const*
valueB *
dtype0*+
_class!
loc:@loss/dense_3_loss/Mean_3*
_output_shapes
: 
µ
>training/RMSprop/gradients/loss/dense_3_loss/Mean_3_grad/ConstConst*
valueB: *
dtype0*+
_class!
loc:@loss/dense_3_loss/Mean_3*
_output_shapes
:
≤
=training/RMSprop/gradients/loss/dense_3_loss/Mean_3_grad/ProdProd@training/RMSprop/gradients/loss/dense_3_loss/Mean_3_grad/Shape_1>training/RMSprop/gradients/loss/dense_3_loss/Mean_3_grad/Const*
	keep_dims( *
T0*

Tidx0*+
_class!
loc:@loss/dense_3_loss/Mean_3*
_output_shapes
: 
Ј
@training/RMSprop/gradients/loss/dense_3_loss/Mean_3_grad/Const_1Const*
valueB: *
dtype0*+
_class!
loc:@loss/dense_3_loss/Mean_3*
_output_shapes
:
ґ
?training/RMSprop/gradients/loss/dense_3_loss/Mean_3_grad/Prod_1Prod@training/RMSprop/gradients/loss/dense_3_loss/Mean_3_grad/Shape_2@training/RMSprop/gradients/loss/dense_3_loss/Mean_3_grad/Const_1*
	keep_dims( *
T0*

Tidx0*+
_class!
loc:@loss/dense_3_loss/Mean_3*
_output_shapes
: 
±
Btraining/RMSprop/gradients/loss/dense_3_loss/Mean_3_grad/Maximum/yConst*
value	B :*
dtype0*+
_class!
loc:@loss/dense_3_loss/Mean_3*
_output_shapes
: 
Ю
@training/RMSprop/gradients/loss/dense_3_loss/Mean_3_grad/MaximumMaximum?training/RMSprop/gradients/loss/dense_3_loss/Mean_3_grad/Prod_1Btraining/RMSprop/gradients/loss/dense_3_loss/Mean_3_grad/Maximum/y*
T0*+
_class!
loc:@loss/dense_3_loss/Mean_3*
_output_shapes
: 
Ь
Atraining/RMSprop/gradients/loss/dense_3_loss/Mean_3_grad/floordivFloorDiv=training/RMSprop/gradients/loss/dense_3_loss/Mean_3_grad/Prod@training/RMSprop/gradients/loss/dense_3_loss/Mean_3_grad/Maximum*
T0*+
_class!
loc:@loss/dense_3_loss/Mean_3*
_output_shapes
: 
е
=training/RMSprop/gradients/loss/dense_3_loss/Mean_3_grad/CastCastAtraining/RMSprop/gradients/loss/dense_3_loss/Mean_3_grad/floordiv*

SrcT0*

DstT0*+
_class!
loc:@loss/dense_3_loss/Mean_3*
_output_shapes
: 
§
@training/RMSprop/gradients/loss/dense_3_loss/Mean_3_grad/truedivRealDiv=training/RMSprop/gradients/loss/dense_3_loss/Mean_3_grad/Tile=training/RMSprop/gradients/loss/dense_3_loss/Mean_3_grad/Cast*
T0*+
_class!
loc:@loss/dense_3_loss/Mean_3*#
_output_shapes
:€€€€€€€€€
¬
?training/RMSprop/gradients/loss/dense_3_loss/truediv_grad/ShapeShapeloss/dense_3_loss/mul*
T0*
out_type0*,
_class"
 loc:@loss/dense_3_loss/truediv*
_output_shapes
:
≤
Atraining/RMSprop/gradients/loss/dense_3_loss/truediv_grad/Shape_1Const*
valueB *
dtype0*,
_class"
 loc:@loss/dense_3_loss/truediv*
_output_shapes
: 
„
Otraining/RMSprop/gradients/loss/dense_3_loss/truediv_grad/BroadcastGradientArgsBroadcastGradientArgs?training/RMSprop/gradients/loss/dense_3_loss/truediv_grad/ShapeAtraining/RMSprop/gradients/loss/dense_3_loss/truediv_grad/Shape_1*
T0*,
_class"
 loc:@loss/dense_3_loss/truediv*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
Д
Atraining/RMSprop/gradients/loss/dense_3_loss/truediv_grad/RealDivRealDiv@training/RMSprop/gradients/loss/dense_3_loss/Mean_3_grad/truedivloss/dense_3_loss/Mean_2*
T0*,
_class"
 loc:@loss/dense_3_loss/truediv*#
_output_shapes
:€€€€€€€€€
∆
=training/RMSprop/gradients/loss/dense_3_loss/truediv_grad/SumSumAtraining/RMSprop/gradients/loss/dense_3_loss/truediv_grad/RealDivOtraining/RMSprop/gradients/loss/dense_3_loss/truediv_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*,
_class"
 loc:@loss/dense_3_loss/truediv*
_output_shapes
:
ґ
Atraining/RMSprop/gradients/loss/dense_3_loss/truediv_grad/ReshapeReshape=training/RMSprop/gradients/loss/dense_3_loss/truediv_grad/Sum?training/RMSprop/gradients/loss/dense_3_loss/truediv_grad/Shape*
T0*
Tshape0*,
_class"
 loc:@loss/dense_3_loss/truediv*#
_output_shapes
:€€€€€€€€€
Ј
=training/RMSprop/gradients/loss/dense_3_loss/truediv_grad/NegNegloss/dense_3_loss/mul*
T0*,
_class"
 loc:@loss/dense_3_loss/truediv*#
_output_shapes
:€€€€€€€€€
Г
Ctraining/RMSprop/gradients/loss/dense_3_loss/truediv_grad/RealDiv_1RealDiv=training/RMSprop/gradients/loss/dense_3_loss/truediv_grad/Negloss/dense_3_loss/Mean_2*
T0*,
_class"
 loc:@loss/dense_3_loss/truediv*#
_output_shapes
:€€€€€€€€€
Й
Ctraining/RMSprop/gradients/loss/dense_3_loss/truediv_grad/RealDiv_2RealDivCtraining/RMSprop/gradients/loss/dense_3_loss/truediv_grad/RealDiv_1loss/dense_3_loss/Mean_2*
T0*,
_class"
 loc:@loss/dense_3_loss/truediv*#
_output_shapes
:€€€€€€€€€
І
=training/RMSprop/gradients/loss/dense_3_loss/truediv_grad/mulMul@training/RMSprop/gradients/loss/dense_3_loss/Mean_3_grad/truedivCtraining/RMSprop/gradients/loss/dense_3_loss/truediv_grad/RealDiv_2*
T0*,
_class"
 loc:@loss/dense_3_loss/truediv*#
_output_shapes
:€€€€€€€€€
∆
?training/RMSprop/gradients/loss/dense_3_loss/truediv_grad/Sum_1Sum=training/RMSprop/gradients/loss/dense_3_loss/truediv_grad/mulQtraining/RMSprop/gradients/loss/dense_3_loss/truediv_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*,
_class"
 loc:@loss/dense_3_loss/truediv*
_output_shapes
:
ѓ
Ctraining/RMSprop/gradients/loss/dense_3_loss/truediv_grad/Reshape_1Reshape?training/RMSprop/gradients/loss/dense_3_loss/truediv_grad/Sum_1Atraining/RMSprop/gradients/loss/dense_3_loss/truediv_grad/Shape_1*
T0*
Tshape0*,
_class"
 loc:@loss/dense_3_loss/truediv*
_output_shapes
: 
љ
;training/RMSprop/gradients/loss/dense_3_loss/mul_grad/ShapeShapeloss/dense_3_loss/Mean_1*
T0*
out_type0*(
_class
loc:@loss/dense_3_loss/mul*
_output_shapes
:
љ
=training/RMSprop/gradients/loss/dense_3_loss/mul_grad/Shape_1Shapedense_3_sample_weights*
T0*
out_type0*(
_class
loc:@loss/dense_3_loss/mul*
_output_shapes
:
«
Ktraining/RMSprop/gradients/loss/dense_3_loss/mul_grad/BroadcastGradientArgsBroadcastGradientArgs;training/RMSprop/gradients/loss/dense_3_loss/mul_grad/Shape=training/RMSprop/gradients/loss/dense_3_loss/mul_grad/Shape_1*
T0*(
_class
loc:@loss/dense_3_loss/mul*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
у
9training/RMSprop/gradients/loss/dense_3_loss/mul_grad/mulMulAtraining/RMSprop/gradients/loss/dense_3_loss/truediv_grad/Reshapedense_3_sample_weights*
T0*(
_class
loc:@loss/dense_3_loss/mul*#
_output_shapes
:€€€€€€€€€
≤
9training/RMSprop/gradients/loss/dense_3_loss/mul_grad/SumSum9training/RMSprop/gradients/loss/dense_3_loss/mul_grad/mulKtraining/RMSprop/gradients/loss/dense_3_loss/mul_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*(
_class
loc:@loss/dense_3_loss/mul*
_output_shapes
:
¶
=training/RMSprop/gradients/loss/dense_3_loss/mul_grad/ReshapeReshape9training/RMSprop/gradients/loss/dense_3_loss/mul_grad/Sum;training/RMSprop/gradients/loss/dense_3_loss/mul_grad/Shape*
T0*
Tshape0*(
_class
loc:@loss/dense_3_loss/mul*#
_output_shapes
:€€€€€€€€€
ч
;training/RMSprop/gradients/loss/dense_3_loss/mul_grad/mul_1Mulloss/dense_3_loss/Mean_1Atraining/RMSprop/gradients/loss/dense_3_loss/truediv_grad/Reshape*
T0*(
_class
loc:@loss/dense_3_loss/mul*#
_output_shapes
:€€€€€€€€€
Є
;training/RMSprop/gradients/loss/dense_3_loss/mul_grad/Sum_1Sum;training/RMSprop/gradients/loss/dense_3_loss/mul_grad/mul_1Mtraining/RMSprop/gradients/loss/dense_3_loss/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*(
_class
loc:@loss/dense_3_loss/mul*
_output_shapes
:
ђ
?training/RMSprop/gradients/loss/dense_3_loss/mul_grad/Reshape_1Reshape;training/RMSprop/gradients/loss/dense_3_loss/mul_grad/Sum_1=training/RMSprop/gradients/loss/dense_3_loss/mul_grad/Shape_1*
T0*
Tshape0*(
_class
loc:@loss/dense_3_loss/mul*#
_output_shapes
:€€€€€€€€€
Ѕ
>training/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/ShapeShapeloss/dense_3_loss/Mean*
T0*
out_type0*+
_class!
loc:@loss/dense_3_loss/Mean_1*
_output_shapes
:
ђ
=training/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/SizeConst*
value	B :*
dtype0*+
_class!
loc:@loss/dense_3_loss/Mean_1*
_output_shapes
: 
ю
<training/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/addAdd*loss/dense_3_loss/Mean_1/reduction_indices=training/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/Size*
T0*+
_class!
loc:@loss/dense_3_loss/Mean_1*
_output_shapes
: 
Х
<training/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/modFloorMod<training/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/add=training/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/Size*
T0*+
_class!
loc:@loss/dense_3_loss/Mean_1*
_output_shapes
: 
Ј
@training/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/Shape_1Const*
valueB: *
dtype0*+
_class!
loc:@loss/dense_3_loss/Mean_1*
_output_shapes
:
≥
Dtraining/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/range/startConst*
value	B : *
dtype0*+
_class!
loc:@loss/dense_3_loss/Mean_1*
_output_shapes
: 
≥
Dtraining/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/range/deltaConst*
value	B :*
dtype0*+
_class!
loc:@loss/dense_3_loss/Mean_1*
_output_shapes
: 
з
>training/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/rangeRangeDtraining/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/range/start=training/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/SizeDtraining/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/range/delta*

Tidx0*+
_class!
loc:@loss/dense_3_loss/Mean_1*
_output_shapes
:
≤
Ctraining/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/Fill/valueConst*
value	B :*
dtype0*+
_class!
loc:@loss/dense_3_loss/Mean_1*
_output_shapes
: 
Ь
=training/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/FillFill@training/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/Shape_1Ctraining/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/Fill/value*
T0*+
_class!
loc:@loss/dense_3_loss/Mean_1*
_output_shapes
: 
Є
Ftraining/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/DynamicStitchDynamicStitch>training/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/range<training/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/mod>training/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/Shape=training/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/Fill*
N*
T0*+
_class!
loc:@loss/dense_3_loss/Mean_1*#
_output_shapes
:€€€€€€€€€
±
Btraining/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/Maximum/yConst*
value	B :*
dtype0*+
_class!
loc:@loss/dense_3_loss/Mean_1*
_output_shapes
: 
≤
@training/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/MaximumMaximumFtraining/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/DynamicStitchBtraining/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/Maximum/y*
T0*+
_class!
loc:@loss/dense_3_loss/Mean_1*#
_output_shapes
:€€€€€€€€€
™
Atraining/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/floordivFloorDiv>training/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/Shape@training/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/Maximum*
T0*+
_class!
loc:@loss/dense_3_loss/Mean_1*#
_output_shapes
:€€€€€€€€€
∞
@training/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/ReshapeReshape=training/RMSprop/gradients/loss/dense_3_loss/mul_grad/ReshapeFtraining/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/DynamicStitch*
T0*
Tshape0*+
_class!
loc:@loss/dense_3_loss/Mean_1*
_output_shapes
:
ђ
=training/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/TileTile@training/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/ReshapeAtraining/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/floordiv*
T0*

Tmultiples0*+
_class!
loc:@loss/dense_3_loss/Mean_1*
_output_shapes
:
√
@training/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/Shape_2Shapeloss/dense_3_loss/Mean*
T0*
out_type0*+
_class!
loc:@loss/dense_3_loss/Mean_1*
_output_shapes
:
≈
@training/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/Shape_3Shapeloss/dense_3_loss/Mean_1*
T0*
out_type0*+
_class!
loc:@loss/dense_3_loss/Mean_1*
_output_shapes
:
µ
>training/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/ConstConst*
valueB: *
dtype0*+
_class!
loc:@loss/dense_3_loss/Mean_1*
_output_shapes
:
≤
=training/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/ProdProd@training/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/Shape_2>training/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/Const*
	keep_dims( *
T0*

Tidx0*+
_class!
loc:@loss/dense_3_loss/Mean_1*
_output_shapes
: 
Ј
@training/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/Const_1Const*
valueB: *
dtype0*+
_class!
loc:@loss/dense_3_loss/Mean_1*
_output_shapes
:
ґ
?training/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/Prod_1Prod@training/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/Shape_3@training/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/Const_1*
	keep_dims( *
T0*

Tidx0*+
_class!
loc:@loss/dense_3_loss/Mean_1*
_output_shapes
: 
≥
Dtraining/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/Maximum_1/yConst*
value	B :*
dtype0*+
_class!
loc:@loss/dense_3_loss/Mean_1*
_output_shapes
: 
Ґ
Btraining/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/Maximum_1Maximum?training/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/Prod_1Dtraining/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/Maximum_1/y*
T0*+
_class!
loc:@loss/dense_3_loss/Mean_1*
_output_shapes
: 
†
Ctraining/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/floordiv_1FloorDiv=training/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/ProdBtraining/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/Maximum_1*
T0*+
_class!
loc:@loss/dense_3_loss/Mean_1*
_output_shapes
: 
з
=training/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/CastCastCtraining/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/floordiv_1*

SrcT0*

DstT0*+
_class!
loc:@loss/dense_3_loss/Mean_1*
_output_shapes
: 
§
@training/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/truedivRealDiv=training/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/Tile=training/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/Cast*
T0*+
_class!
loc:@loss/dense_3_loss/Mean_1*#
_output_shapes
:€€€€€€€€€
њ
<training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/ShapeShapeloss/dense_3_loss/Square*
T0*
out_type0*)
_class
loc:@loss/dense_3_loss/Mean*
_output_shapes
:
®
;training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/SizeConst*
value	B :*
dtype0*)
_class
loc:@loss/dense_3_loss/Mean*
_output_shapes
: 
ф
:training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/addAdd(loss/dense_3_loss/Mean/reduction_indices;training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/Size*
T0*)
_class
loc:@loss/dense_3_loss/Mean*
_output_shapes
: 
Л
:training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/modFloorMod:training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/add;training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/Size*
T0*)
_class
loc:@loss/dense_3_loss/Mean*
_output_shapes
: 
ђ
>training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/Shape_1Const*
valueB *
dtype0*)
_class
loc:@loss/dense_3_loss/Mean*
_output_shapes
: 
ѓ
Btraining/RMSprop/gradients/loss/dense_3_loss/Mean_grad/range/startConst*
value	B : *
dtype0*)
_class
loc:@loss/dense_3_loss/Mean*
_output_shapes
: 
ѓ
Btraining/RMSprop/gradients/loss/dense_3_loss/Mean_grad/range/deltaConst*
value	B :*
dtype0*)
_class
loc:@loss/dense_3_loss/Mean*
_output_shapes
: 
Ё
<training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/rangeRangeBtraining/RMSprop/gradients/loss/dense_3_loss/Mean_grad/range/start;training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/SizeBtraining/RMSprop/gradients/loss/dense_3_loss/Mean_grad/range/delta*

Tidx0*)
_class
loc:@loss/dense_3_loss/Mean*
_output_shapes
:
Ѓ
Atraining/RMSprop/gradients/loss/dense_3_loss/Mean_grad/Fill/valueConst*
value	B :*
dtype0*)
_class
loc:@loss/dense_3_loss/Mean*
_output_shapes
: 
Т
;training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/FillFill>training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/Shape_1Atraining/RMSprop/gradients/loss/dense_3_loss/Mean_grad/Fill/value*
T0*)
_class
loc:@loss/dense_3_loss/Mean*
_output_shapes
: 
ђ
Dtraining/RMSprop/gradients/loss/dense_3_loss/Mean_grad/DynamicStitchDynamicStitch<training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/range:training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/mod<training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/Shape;training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/Fill*
N*
T0*)
_class
loc:@loss/dense_3_loss/Mean*#
_output_shapes
:€€€€€€€€€
≠
@training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/Maximum/yConst*
value	B :*
dtype0*)
_class
loc:@loss/dense_3_loss/Mean*
_output_shapes
: 
™
>training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/MaximumMaximumDtraining/RMSprop/gradients/loss/dense_3_loss/Mean_grad/DynamicStitch@training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/Maximum/y*
T0*)
_class
loc:@loss/dense_3_loss/Mean*#
_output_shapes
:€€€€€€€€€
Щ
?training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/floordivFloorDiv<training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/Shape>training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/Maximum*
T0*)
_class
loc:@loss/dense_3_loss/Mean*
_output_shapes
:
≠
>training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/ReshapeReshape@training/RMSprop/gradients/loss/dense_3_loss/Mean_1_grad/truedivDtraining/RMSprop/gradients/loss/dense_3_loss/Mean_grad/DynamicStitch*
T0*
Tshape0*)
_class
loc:@loss/dense_3_loss/Mean*
_output_shapes
:
Љ
;training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/TileTile>training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/Reshape?training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/floordiv*
T0*

Tmultiples0*)
_class
loc:@loss/dense_3_loss/Mean*0
_output_shapes
:€€€€€€€€€€€€€€€€€€
Ѕ
>training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/Shape_2Shapeloss/dense_3_loss/Square*
T0*
out_type0*)
_class
loc:@loss/dense_3_loss/Mean*
_output_shapes
:
њ
>training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/Shape_3Shapeloss/dense_3_loss/Mean*
T0*
out_type0*)
_class
loc:@loss/dense_3_loss/Mean*
_output_shapes
:
±
<training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/ConstConst*
valueB: *
dtype0*)
_class
loc:@loss/dense_3_loss/Mean*
_output_shapes
:
™
;training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/ProdProd>training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/Shape_2<training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/Const*
	keep_dims( *
T0*

Tidx0*)
_class
loc:@loss/dense_3_loss/Mean*
_output_shapes
: 
≥
>training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/Const_1Const*
valueB: *
dtype0*)
_class
loc:@loss/dense_3_loss/Mean*
_output_shapes
:
Ѓ
=training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/Prod_1Prod>training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/Shape_3>training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/Const_1*
	keep_dims( *
T0*

Tidx0*)
_class
loc:@loss/dense_3_loss/Mean*
_output_shapes
: 
ѓ
Btraining/RMSprop/gradients/loss/dense_3_loss/Mean_grad/Maximum_1/yConst*
value	B :*
dtype0*)
_class
loc:@loss/dense_3_loss/Mean*
_output_shapes
: 
Ъ
@training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/Maximum_1Maximum=training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/Prod_1Btraining/RMSprop/gradients/loss/dense_3_loss/Mean_grad/Maximum_1/y*
T0*)
_class
loc:@loss/dense_3_loss/Mean*
_output_shapes
: 
Ш
Atraining/RMSprop/gradients/loss/dense_3_loss/Mean_grad/floordiv_1FloorDiv;training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/Prod@training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/Maximum_1*
T0*)
_class
loc:@loss/dense_3_loss/Mean*
_output_shapes
: 
б
;training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/CastCastAtraining/RMSprop/gradients/loss/dense_3_loss/Mean_grad/floordiv_1*

SrcT0*

DstT0*)
_class
loc:@loss/dense_3_loss/Mean*
_output_shapes
: 
©
>training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/truedivRealDiv;training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/Tile;training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/Cast*
T0*)
_class
loc:@loss/dense_3_loss/Mean*0
_output_shapes
:€€€€€€€€€€€€€€€€€€
с
>training/RMSprop/gradients/loss/dense_3_loss/Square_grad/mul/xConst?^training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/truediv*
valueB
 *   @*
dtype0*+
_class!
loc:@loss/dense_3_loss/Square*
_output_shapes
: 
В
<training/RMSprop/gradients/loss/dense_3_loss/Square_grad/mulMul>training/RMSprop/gradients/loss/dense_3_loss/Square_grad/mul/xloss/dense_3_loss/sub*
T0*+
_class!
loc:@loss/dense_3_loss/Square*0
_output_shapes
:€€€€€€€€€€€€€€€€€€
Ђ
>training/RMSprop/gradients/loss/dense_3_loss/Square_grad/mul_1Mul>training/RMSprop/gradients/loss/dense_3_loss/Mean_grad/truediv<training/RMSprop/gradients/loss/dense_3_loss/Square_grad/mul*
T0*+
_class!
loc:@loss/dense_3_loss/Square*0
_output_shapes
:€€€€€€€€€€€€€€€€€€
і
;training/RMSprop/gradients/loss/dense_3_loss/sub_grad/ShapeShapedense_3/BiasAdd*
T0*
out_type0*(
_class
loc:@loss/dense_3_loss/sub*
_output_shapes
:
µ
=training/RMSprop/gradients/loss/dense_3_loss/sub_grad/Shape_1Shapedense_3_target*
T0*
out_type0*(
_class
loc:@loss/dense_3_loss/sub*
_output_shapes
:
«
Ktraining/RMSprop/gradients/loss/dense_3_loss/sub_grad/BroadcastGradientArgsBroadcastGradientArgs;training/RMSprop/gradients/loss/dense_3_loss/sub_grad/Shape=training/RMSprop/gradients/loss/dense_3_loss/sub_grad/Shape_1*
T0*(
_class
loc:@loss/dense_3_loss/sub*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
Ј
9training/RMSprop/gradients/loss/dense_3_loss/sub_grad/SumSum>training/RMSprop/gradients/loss/dense_3_loss/Square_grad/mul_1Ktraining/RMSprop/gradients/loss/dense_3_loss/sub_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*(
_class
loc:@loss/dense_3_loss/sub*
_output_shapes
:
™
=training/RMSprop/gradients/loss/dense_3_loss/sub_grad/ReshapeReshape9training/RMSprop/gradients/loss/dense_3_loss/sub_grad/Sum;training/RMSprop/gradients/loss/dense_3_loss/sub_grad/Shape*
T0*
Tshape0*(
_class
loc:@loss/dense_3_loss/sub*'
_output_shapes
:€€€€€€€€€
ї
;training/RMSprop/gradients/loss/dense_3_loss/sub_grad/Sum_1Sum>training/RMSprop/gradients/loss/dense_3_loss/Square_grad/mul_1Mtraining/RMSprop/gradients/loss/dense_3_loss/sub_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*(
_class
loc:@loss/dense_3_loss/sub*
_output_shapes
:
 
9training/RMSprop/gradients/loss/dense_3_loss/sub_grad/NegNeg;training/RMSprop/gradients/loss/dense_3_loss/sub_grad/Sum_1*
T0*(
_class
loc:@loss/dense_3_loss/sub*
_output_shapes
:
Ј
?training/RMSprop/gradients/loss/dense_3_loss/sub_grad/Reshape_1Reshape9training/RMSprop/gradients/loss/dense_3_loss/sub_grad/Neg=training/RMSprop/gradients/loss/dense_3_loss/sub_grad/Shape_1*
T0*
Tshape0*(
_class
loc:@loss/dense_3_loss/sub*0
_output_shapes
:€€€€€€€€€€€€€€€€€€
й
;training/RMSprop/gradients/dense_3/BiasAdd_grad/BiasAddGradBiasAddGrad=training/RMSprop/gradients/loss/dense_3_loss/sub_grad/Reshape*
T0*
data_formatNHWC*"
_class
loc:@dense_3/BiasAdd*
_output_shapes
:
О
5training/RMSprop/gradients/dense_3/MatMul_grad/MatMulMatMul=training/RMSprop/gradients/loss/dense_3_loss/sub_grad/Reshapedense_3/kernel/read*
transpose_a( *
transpose_b(*
T0*!
_class
loc:@dense_3/MatMul*'
_output_shapes
:€€€€€€€€€@
А
7training/RMSprop/gradients/dense_3/MatMul_grad/MatMul_1MatMuldense_2/Relu=training/RMSprop/gradients/loss/dense_3_loss/sub_grad/Reshape*
transpose_a(*
transpose_b( *
T0*!
_class
loc:@dense_3/MatMul*
_output_shapes

:@
ў
5training/RMSprop/gradients/dense_2/Relu_grad/ReluGradReluGrad5training/RMSprop/gradients/dense_3/MatMul_grad/MatMuldense_2/Relu*
T0*
_class
loc:@dense_2/Relu*'
_output_shapes
:€€€€€€€€€@
б
;training/RMSprop/gradients/dense_2/BiasAdd_grad/BiasAddGradBiasAddGrad5training/RMSprop/gradients/dense_2/Relu_grad/ReluGrad*
T0*
data_formatNHWC*"
_class
loc:@dense_2/BiasAdd*
_output_shapes
:@
Ж
5training/RMSprop/gradients/dense_2/MatMul_grad/MatMulMatMul5training/RMSprop/gradients/dense_2/Relu_grad/ReluGraddense_2/kernel/read*
transpose_a( *
transpose_b(*
T0*!
_class
loc:@dense_2/MatMul*'
_output_shapes
:€€€€€€€€€@
ш
7training/RMSprop/gradients/dense_2/MatMul_grad/MatMul_1MatMuldense_1/Relu5training/RMSprop/gradients/dense_2/Relu_grad/ReluGrad*
transpose_a(*
transpose_b( *
T0*!
_class
loc:@dense_2/MatMul*
_output_shapes

:@@
ў
5training/RMSprop/gradients/dense_1/Relu_grad/ReluGradReluGrad5training/RMSprop/gradients/dense_2/MatMul_grad/MatMuldense_1/Relu*
T0*
_class
loc:@dense_1/Relu*'
_output_shapes
:€€€€€€€€€@
б
;training/RMSprop/gradients/dense_1/BiasAdd_grad/BiasAddGradBiasAddGrad5training/RMSprop/gradients/dense_1/Relu_grad/ReluGrad*
T0*
data_formatNHWC*"
_class
loc:@dense_1/BiasAdd*
_output_shapes
:@
Ж
5training/RMSprop/gradients/dense_1/MatMul_grad/MatMulMatMul5training/RMSprop/gradients/dense_1/Relu_grad/ReluGraddense_1/kernel/read*
transpose_a( *
transpose_b(*
T0*!
_class
loc:@dense_1/MatMul*'
_output_shapes
:€€€€€€€€€
щ
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
Н
training/RMSprop/Variable
VariableV2*
shape
:@*
dtype0*
	container *
shared_name *
_output_shapes

:@
Ё
 training/RMSprop/Variable/AssignAssigntraining/RMSprop/Variabletraining/RMSprop/Const*
T0*
validate_shape(*
use_locking(*,
_class"
 loc:@training/RMSprop/Variable*
_output_shapes

:@
Ь
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
З
training/RMSprop/Variable_1
VariableV2*
shape:@*
dtype0*
	container *
shared_name *
_output_shapes
:@
б
"training/RMSprop/Variable_1/AssignAssigntraining/RMSprop/Variable_1training/RMSprop/Const_1*
T0*
validate_shape(*
use_locking(*.
_class$
" loc:@training/RMSprop/Variable_1*
_output_shapes
:@
Ю
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
П
training/RMSprop/Variable_2
VariableV2*
shape
:@@*
dtype0*
	container *
shared_name *
_output_shapes

:@@
е
"training/RMSprop/Variable_2/AssignAssigntraining/RMSprop/Variable_2training/RMSprop/Const_2*
T0*
validate_shape(*
use_locking(*.
_class$
" loc:@training/RMSprop/Variable_2*
_output_shapes

:@@
Ґ
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
З
training/RMSprop/Variable_3
VariableV2*
shape:@*
dtype0*
	container *
shared_name *
_output_shapes
:@
б
"training/RMSprop/Variable_3/AssignAssigntraining/RMSprop/Variable_3training/RMSprop/Const_3*
T0*
validate_shape(*
use_locking(*.
_class$
" loc:@training/RMSprop/Variable_3*
_output_shapes
:@
Ю
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
П
training/RMSprop/Variable_4
VariableV2*
shape
:@*
dtype0*
	container *
shared_name *
_output_shapes

:@
е
"training/RMSprop/Variable_4/AssignAssigntraining/RMSprop/Variable_4training/RMSprop/Const_4*
T0*
validate_shape(*
use_locking(*.
_class$
" loc:@training/RMSprop/Variable_4*
_output_shapes

:@
Ґ
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
З
training/RMSprop/Variable_5
VariableV2*
shape:*
dtype0*
	container *
shared_name *
_output_shapes
:
б
"training/RMSprop/Variable_5/AssignAssigntraining/RMSprop/Variable_5training/RMSprop/Const_5*
T0*
validate_shape(*
use_locking(*.
_class$
" loc:@training/RMSprop/Variable_5*
_output_shapes
:
Ю
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
Є
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
 *  А?*
dtype0*
_output_shapes
: 
f
training/RMSprop/subSubtraining/RMSprop/sub/xRMSprop/rho/read*
T0*
_output_shapes
: 
Г
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
“
training/RMSprop/AssignAssigntraining/RMSprop/Variabletraining/RMSprop/add*
T0*
validate_shape(*
use_locking(*,
_class"
 loc:@training/RMSprop/Variable*
_output_shapes

:@
Р
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
 *  А*
dtype0*
_output_shapes
: 
К
&training/RMSprop/clip_by_value/MinimumMinimumtraining/RMSprop/addtraining/RMSprop/Const_7*
T0*
_output_shapes

:@
Ф
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
 *wћ+2*
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
ј
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
 *  А?*
dtype0*
_output_shapes
: 
j
training/RMSprop/sub_2Subtraining/RMSprop/sub_2/xRMSprop/rho/read*
T0*
_output_shapes
: 
Е
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
÷
training/RMSprop/Assign_2Assigntraining/RMSprop/Variable_1training/RMSprop/add_2*
T0*
validate_shape(*
use_locking(*.
_class$
" loc:@training/RMSprop/Variable_1*
_output_shapes
:@
Р
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
 *  А*
dtype0*
_output_shapes
: 
К
(training/RMSprop/clip_by_value_1/MinimumMinimumtraining/RMSprop/add_2training/RMSprop/Const_9*
T0*
_output_shapes
:@
Ф
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
 *wћ+2*
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
Є
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
 *  А?*
dtype0*
_output_shapes
: 
j
training/RMSprop/sub_4Subtraining/RMSprop/sub_4/xRMSprop/rho/read*
T0*
_output_shapes
: 
Е
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
Џ
training/RMSprop/Assign_4Assigntraining/RMSprop/Variable_2training/RMSprop/add_4*
T0*
validate_shape(*
use_locking(*.
_class$
" loc:@training/RMSprop/Variable_2*
_output_shapes

:@@
Р
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
 *  А*
dtype0*
_output_shapes
: 
П
(training/RMSprop/clip_by_value_2/MinimumMinimumtraining/RMSprop/add_4training/RMSprop/Const_11*
T0*
_output_shapes

:@@
Щ
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
 *wћ+2*
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
ј
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
 *  А?*
dtype0*
_output_shapes
: 
j
training/RMSprop/sub_6Subtraining/RMSprop/sub_6/xRMSprop/rho/read*
T0*
_output_shapes
: 
Е
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
÷
training/RMSprop/Assign_6Assigntraining/RMSprop/Variable_3training/RMSprop/add_6*
T0*
validate_shape(*
use_locking(*.
_class$
" loc:@training/RMSprop/Variable_3*
_output_shapes
:@
С
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
 *  А*
dtype0*
_output_shapes
: 
Л
(training/RMSprop/clip_by_value_3/MinimumMinimumtraining/RMSprop/add_6training/RMSprop/Const_13*
T0*
_output_shapes
:@
Х
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
 *wћ+2*
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
Є
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
 *  А?*
dtype0*
_output_shapes
: 
j
training/RMSprop/sub_8Subtraining/RMSprop/sub_8/xRMSprop/rho/read*
T0*
_output_shapes
: 
Е
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
Џ
training/RMSprop/Assign_8Assigntraining/RMSprop/Variable_4training/RMSprop/add_8*
T0*
validate_shape(*
use_locking(*.
_class$
" loc:@training/RMSprop/Variable_4*
_output_shapes

:@
С
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
 *  А*
dtype0*
_output_shapes
: 
П
(training/RMSprop/clip_by_value_4/MinimumMinimumtraining/RMSprop/add_8training/RMSprop/Const_15*
T0*
_output_shapes

:@
Щ
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
 *wћ+2*
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
ј
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
 *  А?*
dtype0*
_output_shapes
: 
l
training/RMSprop/sub_10Subtraining/RMSprop/sub_10/xRMSprop/rho/read*
T0*
_output_shapes
: 
Е
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
Ў
training/RMSprop/Assign_10Assigntraining/RMSprop/Variable_5training/RMSprop/add_10*
T0*
validate_shape(*
use_locking(*.
_class$
" loc:@training/RMSprop/Variable_5*
_output_shapes
:
С
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
 *  А*
dtype0*
_output_shapes
: 
М
(training/RMSprop/clip_by_value_5/MinimumMinimumtraining/RMSprop/add_10training/RMSprop/Const_17*
T0*
_output_shapes
:
Х
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
 *wћ+2*
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
Ї
training/RMSprop/Assign_11Assigndense_3/biastraining/RMSprop/sub_11*
T0*
validate_shape(*
use_locking(*
_class
loc:@dense_3/bias*
_output_shapes
:
Є
training/group_depsNoOp	^loss/mul#^metrics/mean_absolute_error/Mean_1^training/RMSprop/AssignAdd^training/RMSprop/Assign^training/RMSprop/Assign_1^training/RMSprop/Assign_2^training/RMSprop/Assign_3^training/RMSprop/Assign_4^training/RMSprop/Assign_5^training/RMSprop/Assign_6^training/RMSprop/Assign_7^training/RMSprop/Assign_8^training/RMSprop/Assign_9^training/RMSprop/Assign_10^training/RMSprop/Assign_11
B

group_depsNoOp	^loss/mul#^metrics/mean_absolute_error/Mean_1
Ж
IsVariableInitializedIsVariableInitializeddense_1/kernel*
dtype0*!
_class
loc:@dense_1/kernel*
_output_shapes
: 
Д
IsVariableInitialized_1IsVariableInitializeddense_1/bias*
dtype0*
_class
loc:@dense_1/bias*
_output_shapes
: 
И
IsVariableInitialized_2IsVariableInitializeddense_2/kernel*
dtype0*!
_class
loc:@dense_2/kernel*
_output_shapes
: 
Д
IsVariableInitialized_3IsVariableInitializeddense_2/bias*
dtype0*
_class
loc:@dense_2/bias*
_output_shapes
: 
И
IsVariableInitialized_4IsVariableInitializeddense_3/kernel*
dtype0*!
_class
loc:@dense_3/kernel*
_output_shapes
: 
Д
IsVariableInitialized_5IsVariableInitializeddense_3/bias*
dtype0*
_class
loc:@dense_3/bias*
_output_shapes
: 
А
IsVariableInitialized_6IsVariableInitialized
RMSprop/lr*
dtype0*
_class
loc:@RMSprop/lr*
_output_shapes
: 
В
IsVariableInitialized_7IsVariableInitializedRMSprop/rho*
dtype0*
_class
loc:@RMSprop/rho*
_output_shapes
: 
Ж
IsVariableInitialized_8IsVariableInitializedRMSprop/decay*
dtype0* 
_class
loc:@RMSprop/decay*
_output_shapes
: 
Р
IsVariableInitialized_9IsVariableInitializedRMSprop/iterations*
dtype0	*%
_class
loc:@RMSprop/iterations*
_output_shapes
: 
Я
IsVariableInitialized_10IsVariableInitializedtraining/RMSprop/Variable*
dtype0*,
_class"
 loc:@training/RMSprop/Variable*
_output_shapes
: 
£
IsVariableInitialized_11IsVariableInitializedtraining/RMSprop/Variable_1*
dtype0*.
_class$
" loc:@training/RMSprop/Variable_1*
_output_shapes
: 
£
IsVariableInitialized_12IsVariableInitializedtraining/RMSprop/Variable_2*
dtype0*.
_class$
" loc:@training/RMSprop/Variable_2*
_output_shapes
: 
£
IsVariableInitialized_13IsVariableInitializedtraining/RMSprop/Variable_3*
dtype0*.
_class$
" loc:@training/RMSprop/Variable_3*
_output_shapes
: 
£
IsVariableInitialized_14IsVariableInitializedtraining/RMSprop/Variable_4*
dtype0*.
_class$
" loc:@training/RMSprop/Variable_4*
_output_shapes
: 
£
IsVariableInitialized_15IsVariableInitializedtraining/RMSprop/Variable_5*
dtype0*.
_class$
" loc:@training/RMSprop/Variable_5*
_output_shapes
: 
ќ
initNoOp^dense_1/kernel/Assign^dense_1/bias/Assign^dense_2/kernel/Assign^dense_2/bias/Assign^dense_3/kernel/Assign^dense_3/bias/Assign^RMSprop/lr/Assign^RMSpr