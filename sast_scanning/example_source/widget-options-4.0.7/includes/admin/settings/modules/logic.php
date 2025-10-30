<?php
/**
 * Widget Logic Settings Module
 * Settings > Widget Options :: Display Logic
 *
 * @copyright   Copyright (c) 2016, Jeffrey Carandang
 * @since       3.0
*/

// Exit if accessed directly
if ( ! defined( 'ABSPATH' ) ) exit;

/**
 * Create Card Module for Display Logic Options
 *
 * @since 3.0
 * @global $widget_options
 * @return void
 */
if( !function_exists( 'widgetopts_settings_logic' ) ):
	function widgetopts_settings_logic(){
	    global $widget_options; ?>
		<li class="widgetopts-module-card <?php echo ( isset( $widget_options['logic'] ) && $widget_options['logic'] == 'activate' ) ? 'widgetopts-module-type-enabled' : 'widgetopts-module-type-disabled'; ?>" id="widgetopts-module-card-logic" data-module-id="logic">
			<div class="widgetopts-module-card-content">
				<h2><?php _e( 'Display Logic', 'widget-options' );?></h2>
				<p class="widgetopts-module-desc">
					<?php _e( 'Use WordPress PHP conditional tags to assign each widgets visibility.', 'widget-options' );?>
				</p>

				<div class="widgetopts-module-actions hide-if-no-js">
					<?php if( $widget_options['logic'] == 'activate' ){ ?>
						<button class="button button-secondary widgetopts-toggle-settings"><?php _e( 'Configure Settings', 'widget-options' );?></button>
						<button class="button button-secondary widgetopts-toggle-activation"><?php _e( 'Disable', 'widget-options' );?></button>
					<?php }else{ ?>
						<button class="button button-secondary widgetopts-toggle-settings"><?php _e( 'Learn More', 'widget-options' );?></button>
						<button class="button button-primary widgetopts-toggle-activation"><?php _e( 'Enable', 'widget-options' );?></button>
					<?php } ?>

				</div>

			</div>

			<?php widgetopts_modal_start( $widget_options['logic'] ); ?>
				<span class="dashicons widgetopts-dashicons widgetopts-no-top dashicons-admin-generic"></span>
				<h3 class="widgetopts-modal-header"><?php _e( 'Display Logic', 'widget-options' );?></h3>
				<p>
					<?php _e( 'Display Widget Logic will let you control where you want the widgets to appear using WordPress conditional tags.', 'widget-options' );?>
				</p>
				<p>
					<?php _e( "<strong>Please note</strong> that the display logic you introduce is EVAL'd directly. Anyone who has access to edit widget appearance will have the right to add any code, including malicious and possibly destructive functions. There is an optional filter <code>widget_options_logic_override</code> which you can use to bypass the EVAL with your own code if needed.", 'widget-options' )?>
				</p>
				<table class="form-table widgetopts-settings-section">
					<tr>
						<th scope="row">
							<label for="widgetopts-logic-notice"><?php _e( 'Hide Notice', 'widget-options' );?></label>
						</th>
						<td>
							<input type="checkbox" id="widgetopts-logic-notice" name="logic[notice]" <?php echo ( isset( $widget_options['settings']['logic'] ) ) ? widgetopts_is_checked( $widget_options['settings']['logic'], 'notice' ) : ''; ?> value="1" />
							<label for="widgetopts-logic-notice"><?php _e( 'Disable Notice Toggler', 'widget-options' );?></label>
							<p class="description">
								<?php _e( 'Hide similar filter notice above on each widget display logic feature.', 'widget-options' );?>
							</p>
						</td>
					</tr>
				</table>
			<?php widgetopts_modal_end( $widget_options['logic'] ); ?>

		</li>
	    <?php
	}
	add_action( 'widgetopts_module_cards', 'widgetopts_settings_logic', 60 );
endif;
?>
