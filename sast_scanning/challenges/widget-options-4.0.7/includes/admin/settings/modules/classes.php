<?php
/**
 * Widget Classes Settings Module
 * Settings > Widget Options :: Classes & ID
 *
 * @copyright   Copyright (c) 2016, Jeffrey Carandang
 * @since       3.0
*/

// Exit if accessed directly
if ( ! defined( 'ABSPATH' ) ) exit;

/**
 * Create Card Module for Widget Classes Options
 *
 * @since 3.0
 * @global $widget_options
 * @return void
 */
if( !function_exists( 'widgetopts_settings_classes' ) ):
	function widgetopts_settings_classes(){
		global $widget_options;
		$classes	= ( isset( $widget_options['settings']['classes'] ) ) ? $widget_options['settings']['classes'] : array();
		$classlists = ( isset( $classes['classlists'] ) && is_array( $classes['classlists'] ) ) ? $classes['classlists'] : array();?>
	    <li class="widgetopts-module-card <?php echo ( $widget_options['classes'] == 'activate' ) ? 'widgetopts-module-type-enabled' : 'widgetopts-module-type-disabled'; ?>" id="widgetopts-module-card-classes" data-module-id="classes">
			<div class="widgetopts-module-card-content">
				<h2><?php _e( 'Classes & ID', 'widget-options' );?></h2>
				<p class="widgetopts-module-desc">
					<?php _e( 'Assign custom css classes and ID on each widgets for element targeting.', 'widget-options' );?>
				</p>

				<div class="widgetopts-module-actions hide-if-no-js">
	                <?php if( $widget_options['classes'] == 'activate' ){ ?>
						<button class="button button-secondary widgetopts-toggle-settings"><?php _e( 'Configure Settings', 'widget-options' );?></button>
						<button class="button button-secondary widgetopts-toggle-activation"><?php _e( 'Disable', 'widget-options' );?></button>
					<?php }else{ ?>
						<button class="button button-secondary widgetopts-toggle-settings"><?php _e( 'Learn More', 'widget-options' );?></button>
						<button class="button button-primary widgetopts-toggle-activation"><?php _e( 'Enable', 'widget-options' );?></button>
					<?php } ?>

				</div>

			</div>

			<?php widgetopts_modal_start( $widget_options['classes'] ); ?>
				<span class="dashicons widgetopts-dashicons widgetopts-no-top dashicons-admin-generic"></span>
				<h3 class="widgetopts-modal-header"><?php _e( 'Classes & ID', 'widget-options' );?></h3>
				<p>
					<?php _e( 'Custom alignment widget options will allow you to assign different content alignments for each widgets on specific devices. You can choose whether you want them to be left, right, justify or centered aligned on desktop, tablet or mobile devices.', 'widget-options' );?>
				</p>
				<table class="form-table widgetopts-settings-section">
					<tr>
						<th scope="row">
							<label for="widgetopts-classes-id"><?php _e( 'Show ID Field', 'widget-options' );?></label>
						</th>
						<td>
							<input type="checkbox" id="widgetopts-classes-id" name="classes[id]" <?php echo widgetopts_is_checked( $classes, 'id' ) ?> value="1" />
							<label for="widgetopts-classes-id"><?php _e( 'Enable ID Field', 'widget-options' );?></label>
							<p class="description">
								<?php _e( 'Allow user to add custom ID on each widgets. ', 'widget-options' );?>
							</p>
						</td>
					</tr>
					<tr>
						<th scope="row">
							<label><?php _e( 'Classes Field Type', 'widget-options' );?></label>
						</th>
						<td>
							<label for="widgetopts-classes-class-text">
								<input type="radio" value="text" id="widgetopts-classes-class-text" name="classes[type]" <?php if( isset( $classes['type'] ) && 'text' == $classes['type'] ){ echo 'checked="checked"'; }?> /><?php _e( 'Text Field', 'widget-options' );?>
							</label>&nbsp;&nbsp;

							<label for="widgetopts-classes-class-predefined">
								<input type="radio" value="predefined" id="widgetopts-classes-class-predefined" name="classes[type]" <?php if( isset( $classes['type'] ) && 'predefined' == $classes['type'] ){ echo 'checked="checked"'; }?> /><?php _e( 'Predefined Class Checkboxes', 'widget-options' );?>
							</label>&nbsp;&nbsp;

							<label for="widgetopts-classes-class-both">
								<input type="radio" value="both" id="widgetopts-classes-class-both" name="classes[type]" <?php if( isset( $classes['type'] ) && 'both' == $classes['type'] ){ echo 'checked="checked"'; }?> /><?php _e( 'Both', 'widget-options' );?>
							</label>&nbsp;&nbsp;

							<label for="widgetopts-classes-class-hide">
								<input type="radio" value="hide" id="widgetopts-classes-class-hide" name="classes[type]" <?php if( isset( $classes['type'] ) && 'hide' == $classes['type'] ){ echo 'checked="checked"'; }?> /><?php _e( 'Hide', 'widget-options' );?>
							</label>
							<p class="description">
								<?php _e( 'Select which field type you want to manage each widget classes option.', 'widget-options' );?>
							</p>
						</td>
					</tr>
					<tr>
						<th scope="row">
							<label for="widgetopts-classes-auto"><?php _e( 'Remove .widget Class', 'widget-options' );?></label>
						</th>
						<td>
							<input type="checkbox" id="widgetopts-classes-auto" name="classes[auto]" <?php echo widgetopts_is_checked( $classes, 'auto' ) ?> value="1" />
							<label for="widgetopts-classes-auto"><?php _e( 'Disable Additional Class', 'widget-options' );?></label>
							<p class="description">
								<?php _e( 'Check this box if you want to disable the automatic addition of .widget class', 'widget-options' );?>
							</p>
						</td>
					</tr>
				</table>
				<div class="widgetopts-settings-section">
					<h4><?php _e( 'Predefined Classes', 'widget-options' );?></h4>
					<p><?php _e( 'Set a class lists that you want to be available as pre-choices on the Class/ID Widget Options tab.', 'widget-options' );?></p>

					<div id="opts-predefined-classes">
						<ul>
							<li class="opts-hidden-placeholder"></li>
							<?php
								if( !empty( $classlists ) && is_array( $classlists ) ){
									$classlists = array_unique( $classlists );
									foreach ($classlists as $key => $value) {
										echo '<li><input type="hidden" name="classes[classlists][]" value="'. $value .'" /><span class"opts-li-value">'. $value .'</span> <a href="#" class="opts-remove-class-btn"><span class="dashicons dashicons-dismiss"></span></a></li>';
									}
								}
							?>
						</ul>
					</div>

					<table class="form-table">
						<tbody>
							<tr valign="top">
								<td scope="row" valign="middle">
									<input type="text" class="regular-text code opts-add-class-txtfld" />
									<a href="#" class="opts-add-class-btn widgetopts-add-class-btn"><span class="dashicons dashicons-plus-alt"></span></a><br />
									<small><em><?php _e( 'Note: Click the Plus icon to add the class.', 'widget-options' );?></em></small>
								</td>
							</tr>
						</tbody>
					</table>
				</div>
			<?php widgetopts_modal_end( $widget_options['classes'] ); ?>

		</li>
	    <?php
	}
	add_action( 'widgetopts_module_cards', 'widgetopts_settings_classes', 50 );
endif;
?>
